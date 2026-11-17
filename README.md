# data/

**No raw data is stored in this repository.**

TriviaQA (Joshi, Choi, Weld, Zettlemoyer, ACL 2017) is obtained at run time from Hugging Face:
`datasets.load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")`.
Official page: https://nlp.cs.washington.edu/triviaqa/ . The page states that the University of Washington does not own
the copyright of the questions and documents; no data license was found by me. Use for research only; do not redistribute.

What is stored instead (to be created by later scripts): `data/subset_question_ids.csv` (question ids, split assignment, seeds).
[TEST REQUIRED]: the dataset-access command above has not been executed by me yet.
