# Handwritten_text_recognition

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.
This Neural Network (NN) model recognizes the text contained in the images of segmented words as shown in the illustration below.
As these word-images are smaller than images of complete text-lines, the NN can be kept small and training on the CPU is feasible.

<pre>Sentences in document are converted to words using word_segmentation</pre>

###accuracy

74% of the words from the IAM dataset are correctly recognized by the NN when using vanilla beam search decoding.


<b>OUTPUT:</b>


![Screenshot](./doc/1.png)


<b> 2) ateva_main.py </b>

![Screenshot](./doc/2.png)


