# S15B - FG and Depth Detection 
### Problem Statement : 
Given a background image(bg) and same background image with a foreground(fg) object in it - fgbg , network should predict foreground mask and depth of fg_bg image to assess how far the fg object is w.r.to camera for the given bg. 

### Results
###### Mask Prediction
![Mask_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/Mask_Prediction.png)
###### Mask Ground-Truth
![Mask_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/Ground%20Truth_Mask.png)
###### Depth Prediction
![Depth_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/Depth%20Prediction.png)
###### Depth Ground-Truth
![Depth_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/Ground%20Truth_Depth.png)
###### FGBG
![FGBG](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/FG_BG.png)
###### Link to main ipynb file : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/EVA4P1_S15_Comb_FG_Depth_Prediction.ipynb

### Relevant Points:
- Background images selected were of malls & foreground images selected were of sports players and people at lesiure. Hence curves were complex. 
- BCELoss, SSIM & DiceLoss were tried out. Mask was coming out well for BCELoss and Diceloss whereas Depth was not improving.
- Hence tried out SSIM for both mask and depth. Since, loss for mask was less compared to depth, eventually mask predictions were coming fully dark while depth was coming out well.
- It became evident that same loss function cant be used for both mask & depth. Also, training both together is not helping as one loss is getting priority over another. 
- Since mask was coming out well with diceloss & depth was coming out well with SSIM, it was decided to use different loss functions for each. 
- Transfer learning was put into use to make sure that losses are not contradicting each other.
- Hence initial epochs were trained with diceloss for both depth & mask. As expected, mask improved well while depth was not improving.
- Model parameters thus achieved was saved.
- Then Mask convolution layers were frozen, weights saved so far were used to re-load model and Loss function for Depth was switched  to SSIM. Only Depth layer parameters were made trainable.
- Training was done for few more epochs. This made depth improve significantly while retaining the mask quality as such because we are not making any changes to mask layer weights.
- Thus network was able to achieve good results for both mask & depth. 
- Trained network was then tested against the test-set images (70:30 split was adopted).
- Data strategy adopted for training & testing is detailed below in data-load section. Resizing was done to manage the testing times.
- Network used was a custom one with 8,801,568 parameters. Details are present in Network Section. Mask was predicted with 152,544 parameters. Even with this light weight network, decent results were achieved as listed below (images were complex as mentioned above ).
- Network source Code : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/models/S15_FGDepth_models.py
- Accuracy metric used was IOU. Since this is a pixel comparison between prediction and ground-truth, IOU serves as a good metric. Higher IOU for Pred vs Ground-Truth means model is doing well.
### Data-Load
Data-load strategy used is as follows:
- Total data-set of 400K images were split into 280K training images and 120K testing images.(70:30 split).
- A log file having details of FG_BG and its corresponding background image was created. 
- Images (fg_bg, its bg, ground truth mask & ground truth depth) were selected randomly from this 280K set of images using dataloader  for training .
- To manage training time, images were resized to (64,64), (96,96) and (192,192) from original size of (192,192) and divided into batches.
- Training was done for initial 10 epochs with (64,64) for 180K images.
- Next 2 epochs were training with (96,96) for 60K images.
- Final 1 epoch was training with original size of (192,192).
- Then testing was done for 120K testing images with the weights loaded so far.
- Same strategy was repeated for different combinations. Model was saved after each set of training and weights saved were carried over for further training by loading the model from last saved weights (application of transfer learning).
- Source code for dataloader and transforms can be seen in https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/tree/master/src/dataset

## Network
- Network used was custom one. https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/models/S15_FGDepth_models.py
- Bilinear interpolation with a scale factor of 2 to upsize during depth prediction. Along with this, transpose convolutions were also  employed during upsizing to enable convolution without padding while maintaining size.
![Architecture](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/tree/master/Images)
## Loss Functions
- BCELOSS, SSIM and Diceloss were tried out.
- Eventually Diceloss was used for mask and SSIM for depth.


