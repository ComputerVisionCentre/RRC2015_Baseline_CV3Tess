# RRC2015_Baseline_CV3Tess

This program reproduces the results of the Baseline (CV3 + Tess) reported in the ICDAR2015 Robust Reading Competition.
 
 Created on: May 3, 2015
 Author: Lluis Gomez i Bigorda <lgomez AT cvc.uab.es>


The "Baseline (OpenCV + Tesseract)" makes use of the publicly available pipeline proposed in [1]. Concretely, among the available algorithms in the [OpenCV](http://opencv.org) text module, we use the Class Specific Extremal Regions (CSER) and Exhaustive Search algorithms initially proposed by Neumann and Matas [2,3] along with the perceptual grouping approach of Gomez and Karatzas [4] for text localisation. We apply to the output of this pipeline, the open source [Tesseract](http://code.google.com/p/tesseract-ocr/) OCR engine to perform text recognition.


[1] L. Gomez and D. Karatzas, "Scene text recognition: No country for old men?" in Computer Vision-ACCV 2014 Workshops. Springer, 2014, pp. 157–168.

[2] L. Neumann and J. Matas, "Real-time scene text localization and recognition," in Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on.
IEEE, 2012, pp. 3538–3545.

[3] L. Neumann and J. Matas, "Text Localization in Real-world Images using Efficiently Pruned Exhaustive Search, ICDAR 2011 (Beijing, China)

[4] L. Gomez and D. Karatzas, "Multi-script text extraction from natural scenes," in Document Analysis and Recognition (ICDAR), 2013 12th International Conference on. IEEE, 2013, pp. 467–471.
