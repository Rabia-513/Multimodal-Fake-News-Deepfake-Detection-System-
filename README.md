# Multimodal-Fake-News-Deepfake-Detection-System
Topic:
Multimodal Fake News & Deepfake Detection System 
1. Project Purpose and Overview
The objective of this project is to develop a comprehensive Artificial Intelligence (AI) powered system capable of detecting fake news and deepfakes across various media types: text, images, and videos. The system aims to provide not only accurate classifications (Likely Fake/Likely Real) but also transparent explanations for its predictions. By integrating analyses from different modalities, the system strives for a more robust and reliable detection capability, addressing the complex nature of modern disinformation.

2. Implementation Architecture
The system is designed with a modular architecture, with dedicated components for each modality and a final fusion layer. The entire system is implemented in Python, leveraging powerful deep learning libraries and presented via a user-friendly Gradio web application, designed to run conveniently within a Google Colab environment.
3. Text Analysis Module 
This module focuses on analyzing textual claims and news articles for authenticity.
•	Dataset: The LIAR (Labeling Information Articles with Reason) dataset is utilized for training. This dataset comprises short statements manually fact-checked by PolitiFact, each annotated with a fine-grained truthfulness label (e.g., pants-fire, false, half-true, true). For the purpose of binary classification, these six labels are mapped to two categories: 0 (Fake) and 1 (Real). The system is configured to load this dataset directly from a specified Google Drive path 
•	 
•	Model: A multilingual transformer model, specifically xlm-roberta-base from the Hugging Face transformers library, is used. This pre-trained model is fine-tuned on the binary-labeled LIAR dataset, enabling it to learn patterns indicative of truthful or deceptive language.
•	Functionality:
o	Classification: Predicts whether a given news text or claim is "Likely Fake" or "Likely Real."
o	Confidence Score: Provides a probabilistic measure of the model's certainty in its classification.
o	Explainability (Token Importance): To enhance transparency, the system highlights "Top token importances." This is achieved by observing how much the model's prediction changes when individual tokens (words or sub-word units) in the input text are masked. Tokens causing a significant change are considered highly influential, offering insight into the model's reasoning.
•	Resolved Challenges: Initial implementation issues related to the Hugging Face Trainer expecting a labels column (instead of label_bin) and TrainingArguments keyword changes (evaluation_strategy to eval_strategy) were successfully addressed, ensuring smooth training and data flow.
 

4. Image Analysis Module 📸
This module targets the detection of manipulated or deepfaked images.
•	Dataset: The CASIA v2.0 Image Tampering Detection Dataset serves as the training data. This dataset contains images that are either authentic or have undergone various forms of digital tampering.
•	Technique (ELA): Error Level Analysis (ELA) is applied as a preprocessing step. ELA works by re-saving the image at a specific JPEG quality and then calculating the pixel-wise difference between the original and re-saved versions. Areas with inconsistent error levels often indicate regions that have been tampered with or edited.
•	Model: A ResNet18 convolutional neural network, pre-trained on ImageNet, is fine-tuned on the ELA-processed CASIA dataset. Its final classification layer is adapted for binary fake/real image detection.
•	Explainability (Grad-CAM): To provide visual explanations, Grad-CAM (Gradient-weighted Class Activation Mapping) is implemented. Grad-CAM generates a heatmap overlaid on the ELA image, visually highlighting the specific regions (pixels) that were most influential in the model's decision to classify the image as fake or real.
•	Functionality: Detects potential image manipulation and visually explains the suspicious areas.
•	Resolved Challenges: The system includes a utility to automatically download the Kaggle CASIA dataset, provided the user uploads their kaggle.json API key to the Colab environment.
 
5. Video Analysis Module 
This module extends image detection capabilities to videos, primarily for deepfake detection.
•	Technique: The video analysis operates on a frame-by-frame basis. It samples frames from the input video at a specified frame rate per second (FPS) and then passes each sampled frame through the trained Image Analysis module.
•	Functionality: The "fake" probabilities from individual frames are aggregated (averaged) to yield an overall fake probability score for the entire video. The module also identifies and displays the top N most suspicious frames (those with the highest "fake" probability) along with their respective Grad-CAM explanations, providing visual evidence of potential deepfake artifacts.
 

6. Multimodal Fusion 
The system incorporates a fusion mechanism to combine the insights from the individual text, image, and video analysis modules.
•	Method: A weighted average is used to combine the "fake" probability scores from each available modality. Default weights are assigned (e.g., Text: 0.4, Image: 0.3, Video: 0.2, Audio: 0.1), but these can be adjusted. If only a subset of modalities are provided (e.g., only text and image), the system dynamically re-normalizes the weights for the available modalities.
•	Purpose: This fusion approach aims to provide a more comprehensive and robust final assessment by cross-referencing evidence from different media types, leading to a more reliable overall fake/real classification.
 


7. User Interface (Gradio Web Application):
A Gradio web application serves as the interactive frontend for the system, making it accessible to non-technical users.
•	Dashboard Structure: The interface is organized into distinct tabs:
o	Text Analysis: For pasting and analyzing news text or claims.
o	Image Analysis: For uploading and analyzing images.
o	Video Analysis: For uploading and analyzing video files.
o	Multimodal Fusion: For combining text, image, and video inputs for a holistic analysis.
•	User Interaction: Users can upload content (text, image files, MP4 video files) directly through the browser.
 
•	Outputs: For each analysis, the application displays:
o	A clear Classification ("Likely Fake" / "Likely Real").
o	A numerical Confidence Score.
o	The raw Fake Probability.
o	Relevant Explanations (e.g., token importances for text, Grad-CAM heatmaps for images and video frames).
•	Deployment: The Gradio app is designed to run seamlessly in Google Colab, automatically generating a public, shareable URL for easy access and demonstration.
 
 

8. Conclusion 
This Multimodal Fake News & Deepfake Detection System represents a strong proof-of-concept for integrating various AI techniques to combat disinformation. By combining NLP and computer vision models with explainability features, it provides a valuable tool for content authenticity assessment.
