# SEAT(Stable and Explainable Attention)
![Screenshot from 2023-08-11 22-36-17](https://github.com/lazyCodes7/SEAT/assets/53506835/aec80afc-652f-434a-8db3-1b38c756ff50)
Attention stands out as a pivotal mechanism within the architecture of transformers, playing a foundational role. Nonetheless, like other neural networks, transformers remain susceptible to adversarial attacks. Another issue lies in their inconsistency; minor deviations in random seeds can lead to diverse configurations of the attention module. A potential solution revolves around crafting an attention mechanism that not only retains the essence of traditional attention but also demonstrates resilience against attacks. 

## So how are we doing this then?
Well it goes as follows. Since we want to have an attention that is stable towards attacks we craft one by attacking it and making it stronger:p. Similarly for making it similar to the vanilla attention we try to punish our proposed attention when it fails to mimic the original attention. In short, we define an objective function and train our proposed attention to make it the way we want it to be

## Let's get into the algorithm.
![image](https://github.com/lazyCodes7/SEAT/assets/53506835/7fea49ea-e3c4-41a4-91e7-6dfac86eb425)
It is a whole bunch of maths so let's dumb it down!

- Wo_bar is basically our SEAT and to make it stronger we find a perturbation delta by using the PGD attack
- Then we add the delta to Wo_bar and see how it performs compared to the vanilla attention and penalize it accordingly
- At the same time we punish our attention for not mimicing the predictions and also the vanilla attention

Finally we update our Wo_bar via a standard SGD procedure!


## Why will it work?
To be honest, theoretically if I can say our goal is to find an attention that is stable to perturbation and also retains the way the original attention works. So in a way we are looking for a perturbation so large that even if we add it to our attention we still get a pretty good result and similarly even though our attention is really stable now the way our attention is highlighting the important regions is not so different from the original attention. That's why the objective function has a PGD attack that maximized the perturbation that can be added which in turn will make the attention more stable and for the second effect we try to minimize the dis-similarity with the original attention via two functions.

## Experiments
So to prove this I tried out two things first compared it with the vanilla attention and second compared the performance after adding some perturbation not on the attention but on the embedding. I used a BiLSTM with Attention and trained it for 10 epochs on IMDB(Reviews) Dataset.

### Results on SEAT vs Original Attention
- Jenshen-Shannon-Divergence(Comparing the attention) - 0.00135
- Total-Variation-Distance(Comparing difference in predictions) - 0.318

### Results on SEAT vs Adding perturbation on embedding
- Jenshen-Shannon-Divergence(Comparing the attention) - 0.00136
- Total-Variation-Distance(Comparing difference in predictions) - 0.318

## WIP
- Trying out more models like BERT
- Some more illustrations to prove this better
- Working on more perturbation styles like perturbing the sentence by finding a similar word instead
- Improving the quality of SEAT to match with the results in the paper referred

## For reproducing
```
git clone git@github.com:lazyCodes7/SEAT.git
pip install -r requirements.txt
cd seat
python train.py -d 'cuda'
//yay
```

## Credits
```
@misc{hu2022seat,
      title={SEAT: Stable and Explainable Attention}, 
      author={Lijie Hu and Yixin Liu and Ninghao Liu and Mengdi Huai and Lichao Sun and Di Wang},
      year={2022},
      eprint={2211.13290},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

