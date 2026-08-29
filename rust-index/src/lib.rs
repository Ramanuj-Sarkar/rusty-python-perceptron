//! Fast dataset indexing/query engine for perception training data.
//!
//! Loads a COCO-style annotation file once, builds an in-memory index over
//! object classes, bounding-box sizes and image metadata, and answers
//! filtered queries in milliseconds instead of the seconds a pandas/JSON
//! scan takes on large datasets. Exposed to Python via PyO3 so training
//! and orchestration code can call it like any other Python object.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Deserialize, Clone)]
struct Annotation {
    image_id: u64,
    category_id: u64,
    bbox: [f64; 4], // [x, y, w, h]
}

#[derive(Debug, Deserialize, Clone)]
struct ImageMeta {
    id: u64,
    file_name: String,
    width: u64,
    height: u64,
}

#[derive(Debug, Deserialize, Clone)]
struct Category {
    id: u64,
    name: String,
}

#[derive(Debug, Deserialize)]
struct CocoDataset {
    images: Vec<ImageMeta>,
    annotations: Vec<Annotation>,
    categories: Vec<Category>,
}

/// In-memory index over a COCO-style dataset, queryable by class name and
/// minimum object count.
#[pyclass]
struct DatasetIndex {
    images: HashMap<u64, ImageMeta>,
    annotations_by_image: HashMap<u64, Vec<Annotation>>,
    category_name_to_id: HashMap<String, u64>,
    category_id_to_name: HashMap<u64, String>,
}

#[pymethods]
impl DatasetIndex {
    /// Build the index by loading and parsing a COCO-style JSON annotation file.
    #[new]
    fn new(annotations_path: &str) -> PyResult<Self> {
        let raw = fs::read_to_string(annotations_path)
            .map_err(|e| PyValueError::new_err(format!("failed to read {annotations_path}: {e}")))?;
        let dataset: CocoDataset = serde_json::from_str(&raw)
            .map_err(|e| PyValueError::new_err(format!("failed to parse annotations: {e}")))?;

        let mut images = HashMap::new();
        for img in dataset.images {
            images.insert(img.id, img);
        }

        let mut annotations_by_image: HashMap<u64, Vec<Annotation>> = HashMap::new();
        for ann in dataset.annotations {
            annotations_by_image.entry(ann.image_id).or_default().push(ann);
        }

        let mut category_name_to_id = HashMap::new();
        let mut category_id_to_name = HashMap::new();
        for cat in dataset.categories {
            category_name_to_id.insert(cat.name.clone(), cat.id);
            category_id_to_name.insert(cat.id, cat.name);
        }

        Ok(DatasetIndex {
            images,
            annotations_by_image,
            category_name_to_id,
            category_id_to_name,
        })
    }

    /// Return file names of every image containing at least `min_count`
    /// instances of `class_name`, optionally filtered to objects smaller
    /// than `max_area` pixels (useful for small-object curation).
    #[pyo3(signature = (class_name, min_count=1, max_area=None))]
    fn query_by_class(
        &self,
        class_name: &str,
        min_count: usize,
        max_area: Option<f64>,
    ) -> PyResult<Vec<String>> {
        let category_id = self
            .category_name_to_id
            .get(class_name)
            .ok_or_else(|| PyValueError::new_err(format!("unknown class: {class_name}")))?;

        let mut matches = Vec::new();
        for (image_id, anns) in &self.annotations_by_image {
            let count = anns
                .iter()
                .filter(|a| {
                    a.category_id == *category_id
                        && max_area.map_or(true, |max| a.bbox[2] * a.bbox[3] <= max)
                })
                .count();

            if count >= min_count {
                if let Some(img) = self.images.get(image_id) {
                    matches.push(img.file_name.clone());
                }
            }
        }
        Ok(matches)
    }

    /// Per-class object counts across the whole dataset.
    fn class_counts(&self) -> PyResult<HashMap<String, usize>> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for anns in self.annotations_by_image.values() {
            for ann in anns {
                if let Some(name) = self.category_id_to_name.get(&ann.category_id) {
                    *counts.entry(name.clone()).or_insert(0) += 1;
                }
            }
        }
        Ok(counts)
    }

    fn __len__(&self) -> usize {
        self.images.len()
    }
}

#[pymodule]
fn rust_index(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DatasetIndex>()?;
    Ok(())
}
