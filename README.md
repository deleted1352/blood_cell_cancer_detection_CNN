# Blood Cell Classification: End-to-End ML Pipeline



This project features a Deep Learning system designed to classify five types of white blood cells (Basophils, Eosinophils, Lymphocytes, Monocytes, and Neutrophils). It includes a training pipeline developed in Kaggle and a real-time web interface built with [Streamlit](https://share.streamlit.io/).



Use it now: [LINK](https://bloodcellcancerdetectioncnn-4btz34obfcrc7k8uixstug.streamlit.app/#analysis-result)



###### Technical Highlights



Model Architectures: Explored Transfer Learning using VGG-16 and VGG-19 backbones, freezing base layers to utilize ImageNet weights for custom feature extraction.



Custom Training Head: Implemented a sequential top-level design including a **Flatten** layer, **Dropout(0.5)** for regularization, and a 5-neuron **Dense** layer with **SoftMax** activation.



Utilized **ImageDataGenerator** for real-time data augmentation and handled metadata through a custom **Pandas-based dataframe** mapping system.



###### Engineering Challenges \& Solutions



**Challenge:** Initial predictions in production were inconsistent despite high training accuracy. 

**Solution**: Identified a mismatch in pixel scaling. While standard VGG models often use 0-1 normalization, this pipeline was trained using **MobileNetV2** **preprocessing**, which scales inputs to a \[-1, 1] range. Aligning the Streamlit inference math to this specific function resolved the distribution shift.



**Challenge:** Encountered **ValueError** regarding incompatible layer shapes during deployment.

**Solution**: Debugged the tensor flow to find that while the data generator was set to **244 x 244**, the VGG architecture was strictly compiled for **224 x 224**. Standardized the entire pipeline to **224 x 224** to maintain architectural integrity.



**Challenge**: Confident but incorrect classifications (e.g. Neutrophils identified as Basophils).

**Solution**: WIP.



Contact: ethanshenchen@gmail.com

GitHub: deleted1352

