# Dermatoscopic classification model

Image classification model for multi-class prediction of common dermatologic tumors, based on dermatoscopic 
images. This resnet34 model was trained on the [HAM10000 dataset](https://www.nature.com/articles/sdata2018161), and 
used in the publication [Tschandl P. et al. Nature Medicine 2020](https://www.nature.com/articles/s41591-020-0942-0) 
where we explored human-computer interaction of a classification system.

## Entrypoints
- `app.py`: Gradio web-app for single-image prediction
- `images_predict_extract.py`: Create predictions and feature vectors for images within a folder

For education and research use only. **DO NOT use this to obtain medical advice!**
If you have a skin change in question, seek contact to a health care professional.

## Citation
If you use this model, please consider citing the original work for which it was created:
- Tschandl, P. et al. Human–computer collaboration for skin cancer recognition. Nat Med 26, 1229–1234 (2020). https://doi.org/10.1038/s41591-020-0942-0 
```
@article{Tschandl2020_NatureMedicine,
  author = {Philipp Tschandl and Christoph Rinner and Zoe Apalla and Giuseppe Argenziano and Noel Codella and Allan Halpern and Monika Janda and Aimilios Lallas and Caterina Longo and Josep Malvehy and John Paoli and Susana Puig and Cliff Rosendahl and H. Peter Soyer and Iris Zalaudek and Harald Kittler},
  title = {Human{\textendash}computer collaboration for skin cancer recognition},
  journal = {Nature Medicine},
  volume = {26},
  number = {8},
  year = {2020},
  pages = {1229--1234},
  doi = {10.1038/s41591-020-0942-0},
  url = {https://doi.org/10.1038/s41591-020-0942-0}
}
```