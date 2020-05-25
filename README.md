# S15B - FG and Depth Detection 
### Problem Statement : 
Given a background image(bg) and same background image with a foreground(fg) object in it - fgbg , network should predict foreground mask and depth of fg_bg image to assess how far the fg object is w.r.to camera for the given bg.
### Important Points:
- Background images selected were of malls & foreground images selected were of sports players and people. Hence curves were complex.
- BCELoss, SSIM & DiceLoss were tried out. DiceLoss was used for Masks (FG detection) and SSIM was used for Depth.
- Initial training was done with Diceloss for both Mask & Depth. Mask was coming out well but Depth was not improving. 
- After certain epochs, Mask layers were frozen, weights saved so far were used to load model and Loss function for Depth was switched  to SSIM. Only Depth layer parameters were made trainable.
- Training was done for few more epochs. This made depth improve significantly while retaining the mask quality as such.
- Weights gathered this way were used to test the test-set images (70:30 split was adopted).
- Data strategy adopted for training & testing is detailed below in data-load section.
- Network used was a custom one with 8.8 million parameters. Details are present in Network Section. Even with this light weight network, decent results were achieved as listed below.
### Result snapshot
###### Mask Predctions images
![Mask_Prediction](https://github.com/anilbhatt1/EVA4P1_S15A_Depth_FG_Detection_DataPrep/blob/master/Images_For_ReadMe/BG_Sample10.png)
### Data-Load
Data-load strategy used is as follows:
- Total data-set of 400K images were split into 280K training images and 120K testing images.(70:30 split).
- A log file having details of FG_BG and its corresponding background image was created. 
- Images (fg_bg, its bg, ground truth mask & ground truth depth) were selected by dataloader randomly for training from this 280K set of images.
- To manage training time, images were resized to (64,64), (96,96) and (192,192) from original size of (192,192) and divided into batches.
- Training was done for initial 10 epochs with (64,64) for 180K images.
- Next 2 epochs were training with (96,96) for 60K images.
- Final 1 epoch was training with original size of (192,192).
- Then testing was done for 120K testing images with the weights loaded so far.
- Same strategy was repeated for different size of data. Model was saved after each set of training and weights saved were carried over for further training by loading the model from last saved weights (application of transfer learning).
- Source code for dataloader and trabsforms can be seen in https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/tree/master/src/dataset
## Network
- Network used was 
## Loss Functions
## Strategy Used

