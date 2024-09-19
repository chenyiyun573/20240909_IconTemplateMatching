## Prepare Environment

```
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 20240918 1841 PT
By asking GPT4 if there are some better methods to do this icon finding tasks. It gives answers like:
1. Feature-Based Matching
2. Convolutional Neural Networks (CNN)
3. Siamese Networks for Similarity Learning
4. Deep Metric Learning
5. Template Matching with Deep Features
I do not feel these ways can handle this task perfectly. Actually, I cannot think of much better ways to do these kinds of tasks rather than template matching. 
Now, I plan to end this task by keeping 3 scripts:
Simply finding using template matching.
Add multiple icon size support. 
Add edge only template support to ignore color difference. 

Refine these 3 scripts. 
Git pushed to github as 1.0.0.


### 20240918 1634 PT
The `main.py` is just iterate over all pixel position then try to match the icon pixels. 
Then, `main2.py` added a feature that it will scale the size of icon.png to do template matching. 

The code of main2.py works well in some cases. 

The `main3.py` applied the edge detection based on `main2.py`, it works, but not very good using Gaussian. 


Asking GPT4:
Adjusting for more dynamically scaled icons, particularly when dealing with varying display settings and resolutions, indeed calls for a more nuanced approach than fixed-step scaling.

The `main4.py`:
Instead of linear steps, you might consider logarithmic scaling which better accounts for the broader variations in icon sizes due to UI scaling factors. You can scale exponentially, which may be more natural for the kinds of scaling performed by UI frameworks.

However, main4.py do not have result corresponding at scale 1. 


The `main5.py`:
Another effective method, particularly for significant size variations, is to use image pyramids. Pyramids reduce the resolution of the template and the screenshot progressively to capture different levels of details:
Gaussian Pyramid: You can create a series of progressively smaller images, which are smoothed and downsampled.

The result in this example is too blurry and doesn't have a very good effect.


The `main6.py`:
This one is based on main3.py, since this is the UI screenshot, no need to use Gaussain as edge detection, just to differentiate is enough. 


The `main7.py`:
Using the main2_test.py, I found that the icon2.png works the best to find all the openAI icons in the screenshot, so I added this code into main7.py independently. 

The `main8.py`:
Instead of direct pixel matching, feature-based approaches like SIFT (Scale-Invariant Feature Transform) or ORB (Oriented FAST and Rotated BRIEF) can be used. These algorithms detect and describe local features in images that are invariant to scale and rotation, making them ideal for matching objects at different scales and orientations.