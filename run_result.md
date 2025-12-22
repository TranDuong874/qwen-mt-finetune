Epoch = 1
==================================================
Evaluation Results:
  BLEU:       32.19
  chrF++:     55.45
  COMET:      0.8332
  Perplexity: 5.22
==================================================
{"bleu": 32.185024628144006, "chrf++": 55.450096061382034, "comet": 0.833159854327639, "perplexity": 5.220571441991606}
(main) root@C.29103874:/qwen-mt-fi

Epoch = 2
==================================================
Evaluation Results:
  BLEU:       32.79
  chrF++:     55.54
  COMET:      0.8355
  Perplexity: 8.74
==================================================
{"bleu": 32.79094069257578, "chrf++": 55.542526136260214, "comet": 0.8355095522254705, "perplexity": 8.737906388613354}

COMET Score Distribution (full sample):                                                                                     
----------------------------------------                                                                                    
0.0–0.6:    96 (  1.0%)                                                                                                     
0.6–0.7:   435 (  4.3%)                                                                                                     
0.7–0.8:  2040 ( 20.4%)                                                                                                     
0.8–0.9:  5046 ( 50.5%)                                                                                                     
0.9–1.0:  2383 ( 23.8%)                                                                                                     
----------------------------------------                                                                                    
  -> Saved 96 to train_grpo/grpo_gold/00_very_bad.csv                                                                       
  -> Saved 435 to train_grpo/grpo_gold/01_bad.csv                                                                           
  -> Saved 2040 to train_grpo/grpo_gold/02_medium.csv                                                                       
  -> Saved 5046 to train_grpo/grpo_gold/03_good.csv                                                                         
  -> Saved 2383 to train_grpo/grpo_gold/04_excellent.csv                                                                    
                                                                                                                            
Full dataset saved to train_grpo/grpo_gold/all_scored.csv                                                                   
                                                                                                                            
========================================                                                                                    
Creating final gold set: 5000 samples                                                                                       
Distribution: 70% good, 20% medium, 10% bad                                                                                 
========================================                                                                                    
                                                                                                                            
Final gold set: 5000 samples                                                                                                
  Good (>=0.8):     3500 (70.0%)
  Medium (0.7-0.8):  1000 (20.0%)      
  Bad (<0.7):         500 (10.0%) <- FIX THESE

Saved to:
  train_grpo/grpo_gold/gold_set.csv (full)
  gold_good.csv, gold_medium.csv, gold_bad_TO_FIX.csv (by tier)

Summary Statistics:
count    10000.000000
mean         0.840980
std          0.077966
min          0.265628
25%          0.798336
50%          0.852109
75%          0.897647
max          0.989582
Name: comet, dtype: float64

Done!

Next steps:
1. Review 00_very_bad.csv and 01_bad.csv - fix or remove bad translations
2. Spot check 02_medium.csv for edge cases
3. Use 03_good.csv and 04_excellent.csv as positive examples