# S15B - FG and Depth Detection 
### Problem Statement : 
Given a background image(bg) and same background image with a foreground(fg) object in it - fgbg , network should predict foreground mask and depth of fg_bg image to assess how far the fg object is w.r.to camera for the given bg. 

### Relevant Points:
- Background images selected were of malls & foreground images selected were of sports players and people at lesiure. Hence curves were complex as human limb positions can be of any shape. 
- Before starting with CNN approach, OpenCV-Contour method was tried out as an alternative. But results were far from promising. It was evident that a loss function based approach that keeps improving iteratively is the best option.Hence CNN was employed for the purpose. Source Code for opencv POC that was tried out is : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/colab_versions/EVA4P1_S15_OpenCV_FG_Identification_V1.ipynb
- 400K images were split into 280K train images & 120K test images (70:30 split).
- Network used was a custom one with 8,801,568 parameters. Details are present in Network Section. Mask was predicted with 152,544 parameters. Even with this light weight network, decent results were achieved as listed above (images were complex as mentioned above ).
- Since this is image comparison problem, final convolution layers were extracted and fed to the loss function. Final One-hot prediction layers were not included in network as this was not an image classification problem.
- Network source Code : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/models/S15_FGDepth_models.py
- Accuracy metric used was IOU. Since this is a pixel comparison between prediction and ground-truth, IOU serves as a good metric. Higher IOU for Pred vs Ground-Truth means model is doing well.
- Scheduler used was StepLR with initial LR of 0.01 & LR decaying by a factor of 0.1 for every 2 epochs.
#### Training-1
- BCELoss, SSIM & DiceLoss were tried out. Mask was coming out well for BCELoss and Diceloss whereas Depth was not improving.
- Hence tried out SSIM for both mask and depth. As loss for mask was less compared to depth, eventually mask predictions were coming fully dark while depth was coming out well.
- It became evident that same loss function cant be used for both mask & depth. Also, training both together is not helping as one loss is getting priority over another. 
- Since mask was coming out well with diceloss & depth was coming out well with SSIM, it was decided to use different loss functions for each. 
- Transfer learning was put into use to make sure that losses are not contradicting each other. 
- Log is not available of this training as logging mechansim was not proper & colab stopped displaying beyond a point.
#### Training-2
- Hence initial 10 epochs were trained with diceloss for depth & mask. As expected, mask improved well while depth was not improving.
- Model parameters thus achieved was saved.
- Then Mask convolution layers were frozen, weights saved so far were used to re-load model, Loss function for Depth was switched  to SSIM and loss function for mask was retained as DiceLoss itself. This way only Depth layer parameters were kept trainable. Data strategy adopetd for this training is listed in 'Data-Load' section below.
- Training was done for 10 epochs. This improved depth prediction significantly while mask prediction quality was retained as such because no changes were happening to mask layer weights responsible for mask prediction.
- For next 10 epochs, depth layer was frozen and mask layer was trained. Weights from previous epochs were carried over here. This further improved mask prediction while depth prediction quality didnt deteriorate further because no updates were happening to depth layer weights responsible for depth prediction.
- Thus network was able to achieve good results for both mask & depth. 
- For each epoch, trained network was tested against the test-set images of 80K images. 
- Log is not available of this training also as logging mechansim was not proper & colab stopped displaying beyond a point.
#### Training-3
- After this, 10 epochs were trained with same strategy mentioned above. For each epoch, trained network was tested against the entire test-set images of 120K images. Split of data is as follows:
- 0 - 80k    -> size 64x64      |                   
- 80k - 160k -> size 64x64      |
- 160k -220k -> size 64x64      | -> Training Data
- 220k -260k -> size 96x96      |
- 260k -280k -> size 192x192    | 
- 280k -400k -> size 64x64 -> Test dataset
- First 5 epochs, were with mask layers frozen, ONLY DEPTH was trained. Next 5 epochs were with depth layers frozen, ONLY MASK was trained.
- Results and logs of the same are shown as below.
- Data strategy adopted for training & testing is detailed below in data-load section. Resizing was done to manage the testing times.
- Tranforms used were torch transforms - resize, colour jitter.
###### Mask Prediction
![Mask_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_MP_0527.jpg)
###### Mask Ground-Truth
![Mask_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_MA_0527.jpg)
###### Depth Prediction
![Depth_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_DP_0527.jpg)
###### Depth Ground-Truth
![Depth_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_DA_0527.jpg)
###### FGBG
![FGBG](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_FGBG_0527.jpg)
###### Loss Plots
![Loss_Plots](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_LossPlot_0527.png)
###### Link to main ipynb file : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/EVA4P1_S15_Comb_FG_Depth_Prediction.ipynb
###### Link to Test-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_test_log.txt
###### Link to Train-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S14_train_log.txt

#### Training-4
- Same strategy and data split as in Training-3 was adopted. For each epoch, trained network was tested against the entire test-set images of 120K images. 
- One difference was that albumentation transforms were used. Apart from resizing, following transforms were used:
- FG_BG -> IAAAdditiveGaussianNoise() -> This was used to emulate real-life scenario were noise will be present in cameras
- FG_BG -> RGB Shift                  -> This was used to make training hard by shifting RGB channel values
- FG_BG -> Cutout                     -> Certain portions of FG_BG image was cut-out (25% of overall size) so that network will learn to predict without the features those were cut-out. Also in real-life certain portion of image could get cut-out due to various factors.
- BG    -> IAAAdditiveGaussianNoise() -> This was used to emulate real-life scenario were noise will be present in cameras
- BG    ->  RGB Shift                 -> This was used to make training hard by shifting RGB channel values
- BG    -> Cut-Out was NOT USED as we used cut-out already in FG_BG and adding cut-out randomly to BG also will cause adverse affect to network learning. Network may find it hard to figure out background corresponding to fg_bg.
- Mask  -> This is ground-truth. Hence only resize based on epoch was only used.
- Depth -> This is ground-truth. Hence only resize based on epoch was only used.
- Test Images -> Only resize was used. Wanted to keep all other parameters intact as present in original image.
- Results and logs of the same are shown as below.

### Data-Load
Data-load strategy used is as follows:
- Total data-set of 400K images were split into 280K training images and 120K testing images.(70:30 split).
- A log file having details of FG_BG and its corresponding background image was created. 
- Images (fg_bg, its bg, ground truth mask & ground truth depth) were selected randomly from this 280K set of images using dataloader  for training .
- To manage training time, images were resized to (64,64), (96,96) and (192,192) from original size of (192,192) and divided into cohorts of 120K (for 64x64), 120K (for 96x96) and 40K (for 192,192).
- Training was done for initial 6 epochs with (64,64) for 120K images.
- Next 3 epochs were trained with (96,96) for next 120K images.
- Final 1 epoch was trained with original size of (192,192) for remaining 40K images. These were done keeping Mask Convolution layers frozen.
- Then testing was done for in similar manner for same number of epochs (6+2+1) with Depth convolution layers frozen. 
- Model was saved after 500 batches during training. weights saved were carried over for further training by loading the model from last saved weights (application of transfer learning).
- Source code for dataloader and transforms can be seen in https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/tree/master/src/dataset

## Network
- Network used was custom one. Architecture is as shown below. Total Parameters : 8,801,568
- Mask was predicted with 152,544 parameters.
- Source Code : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/models/S15_FGDepth_models.py
- Bilinear interpolation with a scale factor of 2 to upsize during depth prediction. Along with this, transpose convolutions were also  employed during upsizing to enable convolution without padding while maintaining size.
![Architecture](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/EVA15%20Network.jpg)
## Loss Functions
- BCELOSS, SSIM and Diceloss were tried out.
- For final runs, Diceloss was used for mask and SSIM for depth.
- IOU was used as accuracy metric.
- Source code for test and train loss functions can be see here https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/tree/master/src/losses
## Further improvements Plans
- Go for a deeper network to further improve mask & depth predictions. 
- Use Albumentation data transforms - Normalization.
- Use tensor-board.
- Use One-cycle LR policy.
