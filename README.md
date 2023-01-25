<h1> Age Detection from Image</h1>
The objective of my project entitled “AGE DETECTION FROM IMAGE” is to detect the age of the person by giving the front side face image as input. 


  <h3> Database used : UTKFace </h3>
  For this project the dataset named as UTKFace has been used which was downloaded from Kaggle. The dataset contains images of people with ages from 1 to 116. The name of each image has the first letter as the age of the person whose image it is. Each image is of size 200 X 200 pixels in Width X Height. There are around 23700 images in the dataset on which learning and training is to be performed.
  
  
  <h3>Libraries used : </h3>
  <ul>
    <li> Panda </li>
    <li> Numpy</li>
    <li> OS </li>
    <li> matplotlib.pyplot </li>
    <li> Seaborn </li>
    <li> Tensorflow </li>
   
  </ul>
  
  The dataset of images is loaded and then two list are created one for image path and another for age label. In iterative steps while reading the name of an image, as soon as the compiler finds an underscore (_) it appends the age in the age list. The path of each image is also saved in in the list named image_path. Then the lists named image_path and age_label is converted into dataframe which is a tabular representation of list.
  
  <h3> Spliting dataset for training and testing : </h3>
  The Dataset is splitted into 80% for traing purpose and 20% for testing purpose.
  

  
  <h3 >Model training : </h3>
  The model is trained on 80% of the dataset images. Batch size of images, epochs are defined. The function fit takes all these parameters along with numpy array of age and extracted features of image. Just not to train model again and again, it is saved in a file <b>model_weights.h5</b> on which test images will be executed. Also, the user can input an image to predict results.

  <h3> Some Examples : </h3>
  
  ![image](https://user-images.githubusercontent.com/63863073/214569167-59039521-9115-415f-8f63-9f6cf3b884a1.png)
 
  
![Picture1](https://user-images.githubusercontent.com/63863073/214567907-44679d5b-d85f-4332-bddb-d8c82efbb848.png)
