default: 
  @just --list

build-container tag="latest":
  docker build -f docker/Dockerfile -t lumeny/vjepa:{{tag}} .

run-container:
  docker compose -f docker/docker-compose.yml up -d

run-container-force:
  docker compose -f docker/docker-compose.yml up -d --force-recreate

enter-jepa:
  docker exec -it vjepa bash

test-image:
  docker run -it --rm lumeny/vjepa:latest bash

run-eval:
  python3 -m evals.main --fname configs/evals/vitl16_test.yaml --devices cuda:0 cuda:1

alias e := enter-jepa

alias b := build-container

alias r := run-container
alias rf := run-container-force
