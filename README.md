<h1> Age Detection from Image</h1>
The objective of my project entitled “AGE DETECTION FROM IMAGE” is to detect the age of the person by giving the front side face image as input. 


  <h3> Dataset used : UTKFace </h3>
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
  
<h3 >Saving history and Calulating error : </h3>
The history of the trained model is saved in a JSON file name history.json and also with .pkl extension.
The model errors are also calculated and is saved in a CSV file named loss_and_accuracy.csv

  <h3> Loading Model : </h3>
Finally the model is loaded and images are tested from dataset and user input.
