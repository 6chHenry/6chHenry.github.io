# Detect Multiple Objects

need four output numbers for each object: $(x,y,w,h)$ where w stands for width and h stands for height.

## Sliding Window : Apply a CNN to many different crops of the image,CNN Classifies each crop and object or background.

Questions: We have lots of combinations of possible picture sizes, and it will be a large parameter!

## Region Proposals

e.g. Selective search

Find a small set of boxes that are likely to cover all objects. Often based on heuristics: Look for “blob-like” image regions.

R-CNN: Region-Based CNN

Input Image ---> Regions of Interest(ROI) from a proposal method --> Warped Image Regions (224*224)

--> Forward each region through Convnet --> Class Prediction & Bounding box regression(Predict “transform” to correct the ROI: 4 numbers ($t_x,t_y,t_h,t_w$)

Region proposal: ($p_x,p_y,p_h,p_w$)   Transform:($t_x,t_y,t_h,t_w$) Output box:($b_x,b_y,b_h,b_w$)

Translate relative to box size: $b_x=p_x+p_wt_x \quad b_y=p_y+p_ht_y$

Log-space scale transform: $b_w=p_w\exp{t_w}\quad b_h=p_h\exp{t_h}$

