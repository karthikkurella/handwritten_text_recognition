# Handwritten_text_recognition

Handwritten Text Recognition (HTR) system implemented with TensorFlow (TF) and trained on the IAM off-line HTR dataset.
This Neural Network (NN) model recognizes the text contained in the images of segmented words as shown in the illustration below.
As these word-images are smaller than images of complete text-lines, the NN can be kept small and training on the CPU is feasible.

<pre>Sentences in document are converted to words using word_segmentation</pre>

###accuracy

74% of the words from the IAM dataset are correctly recognized by the NN when using vanilla beam search decoding.

<h1>Steps</h1>
<p>1)Go to the wordsegmentaion/src/ directory and main.py. Take care that the image files are placed directly into the wordsegmentaion/data/d2 and not some subdirectory. Afterwards, go to the src/ directory and run python main.py. The input image and the  output will be placed in the wordsegmentaion/out/.</p>
<p>2)Place the segmented images into the test/ folder and open src/ateva_main1.py and run with python. This analyses the images placed in the test folder and processed with neural network ,output written into Ateva.txt file.</p>
3) In order to train the model use src/main.py and run with python main.py --train, these trained weights will be stored to the model/
<b>TRAINING:</b>
<pre>
 python main.py --train
Init with new values
Epoch: 1
Train NN
Batch: 1 / 500 Loss: 130.354
Batch: 2 / 500 Loss: 66.6619
Batch: 3 / 500 Loss: 36.0154
Batch: 4 / 500 Loss: 24.5898
Batch: 5 / 500 Loss: 20.1845
Batch: 6 / 500 Loss: 19.2857
Batch: 7 / 500 Loss: 18.3493
...

Validate NN
Batch: 1 / 115
Ground truth -> Recognized
[OK] "," -> ","
[ERR:1] "Di" -> "D"
[OK] "," -> ","
[OK] """ -> """
[OK] "he" -> "he"
[OK] "told" -> "told"
[ERR:2] "her" -> "nor"
...
Character error rate: 13.956289%. Word accuracy: 67.721739%.
</pre>
<b>OUTPUT:</b>


![Screenshot](./doc/1.png)


<b> 2) ateva_main.py </b>

![Screenshot](./doc/2.png)


