# Valiants-Algorithm-for-Learning-CNF-formulas

An implementation of Valiant's Algorithm for learning Boolean functions by CNF. The input should be the assignment-label data of a Boolean function. The learner learns a k-CNF formula. k-fold cross-validation is applied to evaluate the learnt model.

Algorithm (assume we want to learn a 3-CNF formula):

  C = {all possible 3-CNF clauses}

  For each assignment x with positive label:
      remove from C all clauses that violates x

  return C
  
  To run this python script, run
  
    python learnByValiant.py [filepath] [kfolds] [arity of CNF]
   
  E.g.,
  
    python learnByValiant.py 3cnf/v30/under_v30_c118_r3.933/sat_00001_k3_v30_c118.cnf_Glucose3_10_500.pkl 10 3
