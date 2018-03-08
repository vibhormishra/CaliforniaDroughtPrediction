# California Seasonal Rainfall Prediction

By: Vibhor Mishra, Prasanna Kumar Srinivasachar



### Motivation: 

As evident by the erratic climate changes recorded and analysed by scientist around the world, climate change is becoming one of the major alarming issue. One such instance of it can be seen in California which suffered droughts at the far end of the scale, like the current one that began in 2012, have roughly doubled over the past century. Even though the findings suggest that the drought is primarily a consequence of natural climate variability, the scientists added that the likelihood of any drought becoming acute is rising because of climate change. These changes are expected to continue in the future and more of the precipitation will likely fall as rain instead of snow. This potential change in weather patterns will exacerbate flood risks and add additional challenges for water supply reliability. Talking specifically about California, it receives most of its annual rainfall during the winter months when storms fueled by moisture from the tropical Pacific impact the state. This past winter was the wettest on record for northern California, resulting in massive floods and over 1 billion dollars in damage. 

In such a scenario, ability to predict the rainfall in the months to come, can really help the government in managing resources and prepare for the possible disaster. Some of the flooding problems associated with dams in northern California could be managed better with more accurate seasonal and sub seasonal forecasts of rainfall. If water managers had a skilled forecast of expected rainfall, then they could change the distribution of water in northern California to be more resilient to large rainfall events. The mitigation process can take weeks to complete, so seasonal forecast lead times are needed for effective mitigation. But given the factors that come into play which affect the climate of a region, makes the problem really hard to solve. Current operational seasonal precipitation guidance from the NOAA Climate Prediction Center has no skill above climatology for northern California and is not presented in a way that is useful for water managers at the California Department of Water Resources. Current seasonal precipitation forecasting relies primarily on teleconnection indices, such as ENSO. However, these indices individually are poorly correlated with northern California winter rainfall. For example, the "Godzilla" El Nino of 2015-2016 was expected to bring very heavy rain to California but no extreme rains appeared in the winter of 2015-16. Last years heavy rains happened when ENSO was closer to a neutral state. Other teleconnections should also have some correlation with California rainfall, but finding the most important connections and how they interact is not a task that can be easily done manually.
The goal for our project is to use the November-averaged atmospheric fields to predict the probability of at least 750 mm of rain in northern California between December and February. The observational record for northern California rainfall only goes back to the early 1920s, which would provide a very limited sample size for machine learning or statistical models. Therefore, we are going to use climate model output from simulations run over the last 1000 years. By using climate model output, we hope to sample better the range of possible combinations of weather patterns and rainfall and fit more complex ML and statistical models.

### Dataset:

We are using the NCAR Last Millennium Ensemble which is a set of CESM climate model runs starting in 850 AD and run through 2005. The dataset has 12 full forcing ensemble members from 5 such models, which use the same forcings but had their initial air temperature fields perturbed by a small random round off of 10-14C. The ensemble members thus capture the internal variability within the model's climate system. 

### Analysis and Implication:

We are using Brier Skill scores to measure the performance of our model. While a Brier score answers the question “how large was the error in the forecast?”, the Brier skill score answers the relative skill of the probabilistic forecast over that of climatology, in terms of predicting whether or not an event occurred. It’s calculated using following formula:

BSS = 1 – BS/BSref

Where: BS = Brier score and BSref = BS for the reference forecast.
A “reference forecast” is usually a long-term forecast or similar statistic. A Brier skill score has a range of -∞ to 1.
* Negative values mean that the forecast is less accurate than a standard forecast.
* 0 = no skill compared to the reference forecast.
* 1 = perfect skill compared to the reference forecast.


Starting with feature extraction part, given we had less positive examples in dataset as compare to negative, we resorted to using different techniques to balance the data to improve performance. Here is the list of ones we experimented with:
   1. **Random Under-sampler:** With this method random samples from the majority class are selected to match up the size of minority class to make the data balance.
   2. **Neighborhood Cleaning Rule:** This method utilizes the one-sided selection principle, but considers more carefully the quality of the data to be removed. 
Among the mentioned two, Neighborhood Cleaning Rule performed better and gave better results on the dataset with most plausible reason for being selective in removing data points by cleaning neighborhoods that misclassify examples belonging to the class of interest. Also, we normalized the data before feeding that to our models.

Coming to models, we experimented with variety of NN architecture models on our dataset, starting from Basic NN with one layer, to deep NN with multiple hidden layers, Deep Belief Network, Siamese Network and finally the ensemble of Deep Neural Networks.

   1. NN with one layer: With tanh as activation, this didn’t perform well as expected. With 1 hidden layer and uniform weights initialization, we got the BSS score of less than 0.11 with varying, inconsistent results, with different learning rates(1.0, 0.1, 0.01). The possible reason being that it couldn’t learn the correlation between features and output well and thus, gave the low scores.

   2. NN with 2 hidden layers: We experimented with different hidden units in each hidden layer, starting from 50-100 to 10-50, we realized the network was overfitting the data too quickly which was leading to bad BSS score. The best BSS score we got with this configuration was -0.15.

   3. DBN: Inspired from “A Deep Hybrid Model for Weather Forecasting” paper, we tried DBN with two layers of RBM, 50-150 each, but to our dismay, it didn’t lead to good results. We tried different activation functions, tanh/Elu/Relu and different configurations, varying the learning rate and hidden units but the max BSS score we could get was -0.44 which is less accurate than the standard forecast. The possible reason is same as above, where the dataset isn’t large enough to try networks with 2 or more hidden layers and isn’t too good at generalizing the correlation between features and output for the unseen data. 

   4. Siamese NN : In a nutshell , siamese NN is a twin shared network which is trained over a distance metric(Manhattan distance here) . The network takes in pairs of inputs, if each pair is from the same class ,label is 1 else 0.
Siamese NN implementation gave us a stable and consistent BSS score of 0.125 which was better than all models tried so far. The architecture used was -
      * 60 input features.
      * Hidden layer 1 - Fully connected layer with 20 hidden units.
      * Hidden layer 2 - Fully connected layer with 10 hidden units.
      * Activation function is sigmoid activation.

   5. Ensemble NN: In this architecture, we used an ensemble of NN with 1 layer (from our first model) with early stopping criteria with the train cross split of 70:30. Here are the results we got:
      * Ensemble = 10 with dropout: 0.8, hidden units: 40, tanH activation function, learning rate of 10^-6  and Adam Optimizer as our optimizer function. BSS score : 0.129324 
      * Ensemble = 25, 100 epochs, dropout = 0.8 , learning rate of 0.001, AdamOptimizer, Dense Layer = 40 and tanh as an activation function. BSS score : 0.129453 
      * Ensemble = 48, 100 epochs, dropout = 0.8, lr = 0.001, AdamOptimizer, Dense Layer = 40,  tanh activation. BSS = 0.130115



### Conclusion:
   While basic neural network didn’t as well as expected, but it did give somewhat better results than the baseline. The ensemble approach worked best amongst all the variations tested. Given that the number of training data points we had, i.e. 4572, we were skeptical about the performance of the neural networks overall. Since the dataset is imbalanced, an ensemble of networks gave better results in this case.
   We tried to achieve the max BS score of 0.16 published on the ramp.studios leaderboard for this problem statement but with all the different approaches, we could go upto 0.13 with ensemble approach. The main reason that we believe is the bad dataset here which is not suitable for applying NN architectures. However, if we had got our hands on raw data points, instead of processed ensembled members data for the time period, we could have drawn more significant insight from the data.

### Future work:
Collect better dataset.
Get features from other dataset based on latitude and longitude and do some sort of transfer learning to have better features overall.

### References :
* Srivastava, Nitish. Improving neural networks with dropout. Master’s thesis, University of Toronto, 2013.
* Bromley, Jane, Bentz, James W, Bottou, Leon, Guyon, ´ Isabelle, LeCun, Yann, Moore, Cliff, Sackinger, Eduard, and Shah, Roopak. Signature verification using a siamese time delay neural network. International Journal of Pattern Recognition and Artificial Intelligence, 7 (04):669–688, 1993
* Mart´ın Abadi et al . Whitepaper.TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems,2015
* T.G. Dietterich. Ensemble methods in machine learning, in: International workshop on multiple classier systems.Springer (2000), pp. 1–15
* Clevert et al, Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs), arxiv 2015
* Guillaume Lemaître, Fernando Nogueira, and Christos K. Aridas. “Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning”. In:Journal of Machine Learning Research 18.17 (2017),pp. 1–5.url:http://jmlr.org/papers/v18/16-365
* "ramp-kits/california_rainfall", GitHub, 2017. [Online]. Available: https://github.com/ramp-kits/california_rainfall. [Accessed: 14- Dec- 2017]
* "ywpkwon/siamese_tf_mnist", GitHub. [Online]. Available: https://github.com/ywpkwon/siamese_tf_mnist. [Accessed: 14- Dec- 2017]
