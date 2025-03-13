
# Nicer Than Humans: How do Large Language Models Behave in the Prisoner's Dilemma? <span style="font-size: 0.7em;">([ArXiv](https://arxiv.org/abs/2406.13605))</span>

This repository contains the code for the paper "Nicer Than Humans: How do Large Language Models Behave in the Prisoner's Dilemma?" accepted for publication at [ICWSM'25](https://www.icwsm.org/2025/index.html).

**Authors**:\
Nicolò Fontana (Politecnico di Milano)
[Scholar](https://scholar.google.com/citations?user=fb1sclAAAAAJ&hl=en&oi=ao)\
Francesco Pierri, (Politecnico di Milano) [Scholar](https://scholar.google.com/citations?user=b17WlbMAAAAJ&hl=en&oi=ao)\
Luca Maria Aiello (IT University of Copenhagen, Pioneer Centre for AI) [Scholar](https://scholar.google.com/citations?user=FIX-7hcAAAAJ&hl=en&oi=ao)

**Abstract**:\
The behavior of Large Language Models (LLMs) as artificial social agents is largely unexplored, and we still lack extensive evidence of how these agents react to simple social stimuli. Testing the behavior of AI agents in classic Game Theory experiments provides a promising theoretical framework for evaluating the norms and values of these agents in archetypal social situations. In this work, we investigate the cooperative behavior of three LLMs (Llama2, Llama3, and GPT3.5) when playing the Iterated Prisoner's Dilemma against random adversaries displaying various levels of hostility. We introduce a systematic methodology to evaluate an LLM's comprehension of the game rules and its capability to parse historical gameplay logs for decision-making. We conducted simulations of games lasting for 100 rounds and analyzed the LLMs' decisions in terms of dimensions defined in the behavioral economics literature. We find that all models tend not to initiate defection but act cautiously, favoring cooperation over defection only when the opponent's defection rate is low. Overall, LLMs behave at least as cooperatively as the typical human player, although our results indicate some substantial differences among models. In particular, Llama2 and GPT3.5 are more cooperative than humans, and especially forgiving and non-retaliatory for opponent defection rates below 30%. More similar to humans, Llama3 exhibits consistently uncooperative and exploitative behavior unless the opponent always cooperates. Our systematic approach to the study of LLMs in game theoretical scenarios is a step towards using these simulations to inform practices of LLM auditing and alignment.

## Reproducibility
The analysis performed in the paper can be reproduced by using the available notebooks.
* [Comprehension questions](https://github.com/NicoloFontana/nicer_than_humans_icwsm25/blob/master/1_comprehension_questions_evaluation.ipynb): run some games and evaluate a given LLM over the same meta-questions presented in the paper.


* [Behavioral analysis](https://github.com/NicoloFontana/nicer_than_humans_icwsm25/blob/master/2_behavioral_analysis.ipynb): run some games and analyze the behaviors of a given LLM.


* [Window size comparison](https://github.com/NicoloFontana/nicer_than_humans_icwsm25/blob/master/window_size_comparison.ipynb): run some games with different window sizes for the history provided to a given LLM and compare the results.

## Data
The data used in the paper is available [here](https://github.com/NicoloFontana/nicer_than_humans_icwsm25/tree/master/relevant_runs_copies).

## Acknowledgements
This work was partially supported by the Italian Ministry of Education ( PRIN grant DEMON prot. 2022BAXSPY) and the European Union (NextGenerationEU project PNRR-PE-AI FAIR).

Nicolò Fontana acknowledges the support from the Danish Data Science Academy through the "DDSA Visit Grant" (Grant ID: 2023-1856) and from Politecnico di Milano through the scholarship "Tesi all'estero a.a. 2023/24-Primo bando".

Luca Maria Aiello acknowledges the support from the Carlsberg Foundation through the COCOONS project (CF21-0432).
