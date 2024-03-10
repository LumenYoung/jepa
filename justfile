default: 
  @just --list

run-eval:
  python3 -m evals.main --fname configs/evals/vitl16_test.yaml --devices cuda:0 cuda:1

alias r:=run-eval
