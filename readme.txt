##Requires docker to run##
- Extract the zip file.
- Execute the following steps in terminal
	- cd into Rishabh_project
	- Execute: docker compose build, takes ~30 mins to  build
	- Execute: docker compose up

- To execute on new data, copy the folder containing image, pointcloud and .json in /test/dataset.

Voila!!!! that's it

screw poses are stored in .json file in /test/dataset 

##Error Handling##

- In order to view GUI window of pointcloud, before executing docker compose up use "sudo xhost+" and in dockerfile give the local host ip.