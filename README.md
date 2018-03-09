# Update for unpacked ROB_SLAM with pcl view repo

## build:
First compile g2o and DBoW in the thirdparty folder. Then compile orb slam.

```bash
    mkdir build
    cd build
    cmake ..
    make -j
```

## Run with Orbbec astra
Extract the vocabulary file under Vocabulary folder

under your orb_slam2_astra folder, run this command:

```bash
./bin/rgbd_astra ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/astra.yaml

```
