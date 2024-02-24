# Master-Thesis
Evaluating 3D Image Classification Using Single RGB-D Images

This paper aims to improve conventional RGB image classification systems through
the simple addition of depth information derived from the RGB images themselves.
It builds upon the SynthNet RGB image classification pipeline, which achieved
state-of-the-art results on the VisDA dataset, by integrating monocular estimated
depth channels. Specifically, it attempts to improve Acc@1 on the Topex dataset,
a complex machine parts dataset defined by a significant synthetic to real domain
gap. Utilizing the latest in transfer learning strategies, vision transformer and
monocular depth estimation models, the study devises strategies for data augmentation,
model adaptation, and the application of freezing schedules. It also
conducts a thorough comparison of depth data integration techniques, compares
the performance of monocular estimated versus CAD rendered depth data and
provides a detailed error analysis of the best performing models. The findings
demonstrate a significant enhancement in classification accuracy, culminating in
a model that achieves an Acc@1 of 69.70%. The applied methods offer practical
solutions to industrial challenges and blueprints for integrating depth data into
cutting edge classification models.
