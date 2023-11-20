# TOILC - Trading Optimizer under Information Leakage Constraints


This is a linear optimization solver for the problem of maximizing volume in the financial markets while bounding the leakage of information incurred when trading. 

This software implements a model developed in the paper *Defining and Controlling Information Leakage in US Equities Trading*,  which is set to appear in [PETS 2024](https://petsymposium.org/cfp24.php), and is freely available as a preprint [here](https://eprint.iacr.org/2023/971).


## Dependencies

* You may run the script inside a docker container, using the Dockerfile provided.
* If you would like to run it without using Docker, you will need to have Python 3 installed and the packages matplotlib, numpy and cvxpy. These may be installed via pip by 

```
pip install matplotlib numpy cvxpy
```

## Executing the script

* To run the script, you just need to navigate to the main directory  of the repository and type in the terminal or command prompt

```
python3 toilc.py [COMMANDS]
```

You may also use the solver as a module, instead. A thorough description of the commands can be found in the [user guide](UserGuide.pdf).

If you would like to run it using the [Dockerfile](Dockerfile) provided, build the docker container and run it with the ```bash``` command, i.e.,

```
docker build -t toilc .
docker run -it toilc bash
```

Then, you can run it as above. Note that, if you run the docker container without the ```bash``` command, it will execute the experiments in [experiments.py](experiments.py).

## Experiments


You may check that the solver is working by running some experiments.

```
python3 experiments.py
```

They will also execute if you start the docker container without overriding the default command.

These experiments will recreate figures 6-8 from the paper  *Defining and Controlling Information Leakage in US Equities Trading* in the PETS version, which are the equivalent to the 2<sup>nd</sup>, 3<sup>rd</sup>
 and 4<sup>th</sup> figures in Section 6.1 in the [preprint](https://eprint.iacr.org/2023/971).
 
 All the charts from these experiments will be saved in the folder ```graphs```.

### Experiment 1: Standard parameters
First, we run the solver with the standard parameters of the solver as per the user guide (which can be found in the file [UserGuide.pdf](Userguide.pdf), in which the script generates a distribution for $X$ by sampling from a Gaussian distribution.
Notice that, due to the distribution of $X$ being generated from sampling, there might be some slight variation in the numbers below.

In this first experiment, we run the solver twice, with the objective of exemplifying how Alice's knowledge may change the results of the solver. First, we run it for a scenario where Alice knows $X$ exactly.  This is the equivalent of running

```bash
python3 toilc.py 
```

The terminal should print results similar to these, and the graph should be saved in the file ```Experiment1_PerfectKnowledge.pdf```.
```bash
Results when Alice has perfect knowledge about X:
Expected Value of X: 25.00103180061908
Expected Value of X~: 26.203202644056816
Expected Value of Alice's actions: 1.281370457960506
Variance of Alice's actions: 3.728314312128721
Cutoff point for the left tail: 0
Cutoff point for the right tail: 38
```

Next, we obtain the solution for the situation where Alice does not have any information about the realization of $X$. This is obtained by setting the variable ```noise``` to 1

```bash
python3 toilc.py -noise 1
```

The terminal should print results similar to the ones below, and the graph should be saved in the file ```Experiment1_NoKnowledge.pdf```.

```bash
Results when Alice is completely ignorant about X:
Expected Value of X: 24.996813999044203
Expected Value of X~: 25.520627049539456
Expected Value of Alice's actions: 0.5246143709670591
Variance of Alice's actions: 1.3339297161602526
Cutoff point for the left tail: 0
Cutoff point for the right tail: 38
```

As it can be seen from the results above, the expected value of Alice's actions decreases significantly in a scenario in which she is ignorant about the value of $X$.


### Experiment 2: Figure 6 from the paper
In this experiment, we recreate figure 6 from the paper, which is based on  Q1 2023 SPY data. This data is in the file [experimentData.txt](experimentData.txt), and it will be also used in Experiment 3.

We obtain the results for when Alice has complete knowledge about $X$, using $e^\epsilon=2$ and $\delta=0$. This is the equivalent of typing in the terminal/command prompt:

```bash
python3 toilc.py -distX experimentData.txt -eeps 2 -delta 0 
```

The terminal should print the results below, and the resulting chart can be fount in the file ```Experiment2_Figure6.pdf```. This graph should be similar to the corresponding figure in the paper.

```bash
Results for Figure 6:
Expected Value of X: 100.92192635751269
Expected Value of X~: 111.55908268918354
Expected Value of Alice's actions: 10.637174331670838
Variance of Alice's actions: 5.082777430577857
Cutoff point for the left tail: 0
Cutoff point for the right tail: 201
```

#### Experiment 3: Figures 7 and 8 from the paper
Finally, we explore the situation where Alice is ignorant of $X$, again using Q1 2023 SPY data. 

First, we run the same parameters, only changing the value for ```noise```. This is the equivalent of executing

```bash
python3 toilc.py -distX experimentData.txt -eeps 2 -delta 0 -noise 1
```

The terminal should print the values below, and the resulting chart can be fount in the file ```Experiment3_Figure7.pdf```. 

```bash
Results for Figure 7:
Expected Value of X: 100.9219263575127
Expected Value of X~: 100.92652339632725
Expected Value of Alice's actions: 0.004601255128675394
Variance of Alice's actions: 0.03972105228755908
Cutoff point for the left tail: 0
Cutoff point for the right tail: 201
```

As the real-life distribution is not as smooth as the one from example 1, the lack of knowledge from Alice causes any substantial action to violate the information leakage bounds, which explains the extremely small value of the expectation of Alice's actions.

This can be remedied by ignoring the bounds in the tails of the distribution. To recover Figure 8 from the paper, we set $\delta=0.15$, $\delta_L=0.15$ and $m=1.5$, i.e.

```bash
python3 toilc.py -distX experimentData.txt -eeps 2 -delta 0 -noise 1 \
 -delta 0.15 -deltaLeft 0.15 -tailMult 1.5
```

With these parameters, the solver will ignore the privacy bounds for the first and last 15% of the distribution, and allow Alice's actions to increase the probability mass up to 50% in these regions.

Finally, the terminal should print the values below, and the resulting graph should be saved in the file ```Experiment3_Figure8.pdf```. 


```bash
Results for Figure 8:
Expected Value of X: 100.9219263575127
Expected Value of X~: 107.24951895984363
Expected Value of Alice's actions: 6.332062020764237
Variance of Alice's actions: 3.6049587045621108
Cutoff point for the left tail: 84
Cutoff point for the right tail: 121
```

As it can be seen, ignoring the bounds in the tails of the distribution greatly improved the expectation of Alice's actions.

## Author

Arthur Américo (arthuramerico@gmail.com)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

This project was commissioned by [Proof Trading](www.prooftrading.com), as part of a research project which culminated in the article  [*Defining and Controlling Information Leakage in US Equities Trading*](https://eprint.iacr.org/2023/971). The author of this solver received great inspiration, help and feedback from his paper co-authors Allison Bishop, Paul Cesaretti, Garrison Grogan, Adam McKoy, Robert Moss, Lisa Oakley, Marcel Ribeiro, and Mohammad Shokri.