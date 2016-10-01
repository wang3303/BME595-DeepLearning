# API (img2obj)

* [nil] train()

*In `train()`, Mini-batch & metatable implemented*

* [string] forward([3x32x32 ByteTensor] img)

* [nil] view([3x32x32 ByteTensor] img) -- view the image and prediction

* [nil] cam([int] /idx/) -- fetch images from the camera

*In `cam()`, it will continuously read 1000 frames from the camera before stopping*
