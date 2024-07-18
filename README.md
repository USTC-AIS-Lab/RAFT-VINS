# RAFT-VINS
**RAFT-VINS** is based on RAFT and VINS-Mono

Evaluate in UMA-VI dataset
<p align='center'>
    <img src="./img/uma.gif" alt="drawing" width="800"/>
</p>

Evaluate in self-collected dataset
<p align='center'>
    <img src="./img/self.gif" alt="drawing" width="800"/>
</p>

## 1. Dependices
TensorRT-8.4.1.5
Eigen
ceres-solver

## 2. Build
Clone the repository and catkin_make:

```
cd ~/catkin_ws/src
git clone https://github.com/xixishui1999/RAFT-VINS.git
cd ../
catkin_make -j
source ~/catkin_ws/devel/setup.bash
```

If you only want to test the feature tracking
```
git checkout raftcpp
mkdir build
cd build
cmake ..
make -j
```

## 3. Run
We recommend using the [UMA-VI](https://mapir.isa.uma.es/mapirwebsite/?p=2108) dataset.
```
roslaunch vins_estimator uma.launch
rosbag play third-floor-csc2.bag
```

## 4. Acknowledgements
Thanks for the authors of [RAFT](https://github.com/princeton-vl/RAFT), [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) and [RAFT_CPP](https://github.com/chenjianqu/RAFT_CPP).