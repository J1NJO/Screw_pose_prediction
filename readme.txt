##Requires docker to run##
- Execute the following steps in terminal
	- cd into Screw_pose_prediction_folder
	- Edit the docker yaml file and edit the address of the folder location
	- Execute: docker compose build, takes ~30 mins to  build
	- Execute: docker compose up

- To execute on new data, copy the folder containing image, pointcloud and .json in /test/dataset.

Voila!!!! that's it

screw poses are stored in .json file in /test/dataset 

##Error Handling##

- In order to view GUI window of pointcloud, before executing docker compose up use "sudo xhost+" and in dockerfile give the local host ip.
