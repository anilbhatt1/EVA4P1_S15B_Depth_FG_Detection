# S15B - FG and Depth Detection 
### Problem Statement : 
Given a background image(bg) and same background image with a foreground(fg) object in it - fgbg , network should predict foreground mask and depth of fg_bg image to assess how far the fg object is w.r.to camera for the given bg. 

### Relevant Points:
- Background images selected were of malls & foreground images selected were of sports players and people at lesiure. Hence curves were complex as human limb positions can be of any shape. 
- Before starting with CNN approach, OpenCV-Contour method was tried out as an alternative. But results were far from promising. It was evident that a loss function based approach that keeps improving iteratively is the best option.Hence CNN was employed for the purpose. Source Code for opencv POC that was tried out is : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/colab_versions/EVA4P1_S15_OpenCV_FG_Identification_V1.ipynb
- 400K images were split into 280K train images & 120K test images (70:30 split).More details below under 'Data-Load' section.
- Network used was a custom one with 8,801,568 parameters. Details are present in 'Network' Section. Mask was predicted with 152,544 parameters. Even with this light weight network, decent results were achieved as listed above (images were complex as mentioned above ).
- Since this is image comparison problem, final convolution layers were extracted and fed to the loss function. One-hot prediction layers were not included in network as this was not an image classification problem.
- Network source Code : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/models/S15_FGDepth_models.py
- Accuracy metric used was IOU. Since this was a pixel comparison between prediction and ground-truth, IOU serves as a good metric. Higher IOU for Pred vs Ground-Truth means model is doing well.
- Scheduler used was StepLR with initial LR of 0.01 & LR decaying by a factor of 0.1 for every 2 epochs.
- Weights were saved at each epoch (per 500 batches) and these weights were carried over for further training by loading the model from last saved weights (application of transfer learning).
- Training was done in a progressive manner. Initial trainings (say Training-1) were done with basic set-up. As training progresses, more items were tried out which is listed under relevant section. As mentioned in above point, weights saved at each training phase were used for training subsequent phase.
- Results displayed below are from test epochs.
#### Training-1
- BCELoss, SSIM & DiceLoss were tried out. Mask was coming out well for BCELoss and Diceloss whereas Depth was not improving.
- Hence tried out SSIM for both mask and depth. As loss for mask was less compared to depth, eventually mask predictions were coming fully dark while depth was coming out well.
- It became evident that same loss function cant be used for both mask & depth. Also, training both together was not helping as one loss was getting priority over another. 
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
- Results and logs of the same are shown as below. For display purpose after 10 epochs, a test-loader of batch size 8 from 10,000 images were created (out of 120K test images). Below displayed images are from this test_set.
- Data strategy adopted for training & testing is detailed below in data-load section. Resizing was done to manage the testing times.
- Tranforms used were torch transforms - resize, colour jitter.
- Logs and resulting images were saved to gdrive location. Logs were captured in txt files and images as jpg.
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
###### Loss Function source-code:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Test.py
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Train.py
#### Training-4
- Same strategy and data split as in Training-3 was adopted. For each epoch, trained network was tested against the entire test-set images of 120K images. 
- One improvement was that albumentation transforms were used. Apart from resizing, following transforms were used:
- FG_BG -> IAAAdditiveGaussianNoise() -> This was used to emulate real-life scenario were noise will be present in cameras
- FG_BG -> RGB Shift                  -> This was used to make training hard by shifting RGB channel values
- FG_BG -> Cutout                     -> Certain portions of FG_BG image was cut-out (25% of overall size) so that network will learn to predict without the features those were cut-out. Also in real-life certain portion of image could get cut-out due to various factors.
- BG    -> IAAAdditiveGaussianNoise() -> This was used to emulate real-life scenario were noise will be present in cameras
- BG    ->  RGB Shift                 -> This was used to make training hard by shifting RGB channel values
- BG    -> Cut-Out was NOT USED as we used cut-out already in FG_BG and adding cut-out randomly to BG also will cause adverse affect to network learning. Network may find it hard to figure out background corresponding to fg_bg.
- Mask  -> This is ground-truth. Hence only resize based on respective epoch/data strategy was used.
- Depth -> This is ground-truth. Hence only resize based on respective epoch/data strategy was used.
- Test Images -> Only resize was used. Wanted to keep all other parameters intact as present in original image.
- Results and logs of the same are shown as below. Mask prediction results are seen as going down.
- For display purpose after 10 epochs, a test-loader of batch size 8 from 10,000 images were created (out of 120K test images). Below displayed images are from this test_set.
###### Mask Prediction
![Mask_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_MP_0530.jpg)
###### Mask Ground-Truth
![Mask_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_MA_0530.jpg)
###### Depth Prediction
![Depth_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_DP_0530.jpg)
###### Depth Ground-Truth
![Depth_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_DA_0530.jpg)
###### FGBG
![FGBG](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_FGBG_0530.jpg)
###### Loss Plots
![Loss_Plots](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_LossPlot_0530.png)
###### Link to main ipynb file : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/EVA4P1_S15_Comb_FG_Depth_Prediction_Alb_Transforms(no_norm).ipynb
###### Link to Test-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_test_log.txt
###### Link to Train-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S17_train_log.txt
###### Loss Function source-code:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Test.py
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Train.py
#### Training-5
- Instead of DiceLoss(), BCELOSS() was used for Mask. SSIM was retained for Depth. Loss functions were modified to use BCELOSS for Mask.
- Training was done 4 epochs. First 2 epochs with mask layer frozen (trained for depth parameters) & next 2 epochs with depth layer frozen(trained for mask parameters). For each a test-set of 40K images were tested.
- 0 - 80k    -> size 64x64      |  -> Training Data                 
- 80k - 160k -> size 64x64      |
- 360k -400k -> size 64x64 -> Test dataset
- All other strategies remains same as from Training-4
- Results and logs of the same are shown as below. Mask prediction results improved really well.
- Combination gave promising results with best values of MaskIOU-0.85170 & DepthIOU-0.71749
###### Mask Prediction
![Mask_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_MP_0531.jpg)
###### Mask Ground-Truth
![Mask_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_MA_0531.jpg)
###### Depth Prediction
![Depth_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_DP_0531.jpg)
###### Depth Ground-Truth
![Depth_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_DA_0531.jpg)
###### FGBG
![FGBG](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_FGBG_0531.jpg)
###### Loss Plots
![Loss_Plots](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_LossPlot_0531.png)
###### Link to main ipynb file : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/EVA4P1_S15_Comb_FG_Depth_Prediction_BCE_SSIM.ipynb	
###### Link to Test-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_test_log.txt
###### Link to Train-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S19_train_log.txt
###### Loss Function source-code:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Test1.py
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Train1.py
#### Training-6
- Included normalization into albumentation transforms. Mean and stddev values arrived at during S15-A was used for FG_BG, BG, mask and depth.
- Mask Loss was getting huge even while testing with 1000 images. Depth loss was doing fine. Was not able to fix the loss explosion problem with mask losses. 
- Training and test losses customized to unnormalize and save the images in gdrive is as below:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Train2.py
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Test2.py
- Link to ipynb file where mask output and mask input was unnormalized and fed to loss function is as below. Mask loss was not going high but mask predictions were bad.
https://github.com/anilbhatt1/EVA4P1_S15B_Colab_Notebook_Versions/blob/master/Colab_Versions/EVA4P1_S15_Comb_FG_Depth_Prediction_V20_Normalization.ipynb
- Link to ipynb file where normalized images and predictions were directly fed to loss function is as below. Mask loss can be seen as getting exploded.
https://github.com/anilbhatt1/EVA4P1_S15B_Colab_Notebook_Versions/blob/master/Colab_Versions/EVA4P1_S15_Comb_FG_Depth_Prediction_V21_Normalization.ipynb
#### Training-7
- Modified loss function to fix the mask loss explosion problem. Mask loss was still coming as negative however explosion problem git resolved. Also, IOU measure was coming good. Best results from test MaskIOU-0.84703 DepthIOU-0.78924
- Loss functions were modified in such a way that output from model & input were unnormalized before loss calculations. 
- Link to modified loss functions are as below:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Test3.py
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Train3.py
- Training was done for 4 epochs. 1st & 3rd epoch with mask layers frozen while 2nd and 4th epoch with depth layers frozen. Each epoch were having 80K training images and 10K testing images.
######  Link to ipynb file:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/EVA4P1_S15_Comb_FG_Depth_Prediction_LossFn_Modified_for_Normalization.ipynb
###### Link to Test-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_test_log.txt
###### Link to Train-Logs :
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_train_log.txt
###### Mask Prediction
![Mask_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_MP_0605.jpg)
###### Mask Ground-Truth
![Mask_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_MA_0605.jpg)
###### Depth Prediction
![Depth_Prediction](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_DP_0605.jpg)
###### Depth Ground-Truth
![Depth_GT](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_DA_0605.jpg)
###### FGBG
![FGBG](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_FGBG_0605.jpg)
###### Loss Plots
![Loss_Plots](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/S22_LossPlot_0605.png)
#### Training-8
- Enabled tensor board to capture few scalars.
- Link to modified loss functions are as below:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Test4.py
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/losses/Train4.py
######  Link to ipynb file:
https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/EVA4P1_S15_Comb_FG_Depth_Prediction_TensorBoard.ipynb
### Data-Load
Data-load strategy used is as follows:
- Total data-set of 400K images were split into 280K training images and 120K testing images.(70:30 split).
- A log file having details of FG_BG and its corresponding background image was created. 
- Images (fg_bg, its bg, ground truth mask & ground truth depth) were selected randomly from this 280K set of images using dataloader  for training .
- To manage training time, images were resized to (64,64), (96,96) and (192,192) from original size of (192,192) and divided into cohorts of 80K, 80K, 60K (for 64x64), 40K (for 96x96) and 20K (for 192,192).
- Whole 120K test images were resized to (64,64).
- Weights were smoothly transferred between one size to another size (transfer learning).
- Source code for dataloader and transforms can be seen in https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/tree/master/src/dataset

## Network
- Network used was custom one. Architecture is as shown below. Total Parameters : 8,801,568
- Mask was predicted with 152,544 parameters.
- Source Code : https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/src/models/S15_FGDepth_models.py
- Bilinear interpolation with a scale factor of 2 was used to upsize during depth prediction. Along with this, transpose convolutions were also  employed during upsizing to enable convolution without padding while maintaining size.
![Architecture](https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/blob/master/Images/EVA15%20Network.jpg)
## Loss Functions
- BCELOSS, SSIM and Diceloss were tried out.
- For final runs, BCELOSS was used for mask and SSIM for depth.
- IOU was used as accuracy metric.
- Source code for test and train loss functions can be see here https://github.com/anilbhatt1/EVA4P1_S15B_Depth_FG_Detection/tree/master/src/losses
## Further improvements Plans
- Go for a deeper network to further improve mask & depth predictions. 
- Use Albumentation data transforms - Normalization. (Refer Training-6, tried normalization but not working fine)
- Use tensor-board. (Refer Training-8, enabled tensorboard for scalars. Need to capture images and other parameters)
- Use One-cycle LR policy.
