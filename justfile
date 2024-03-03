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


alias e := enter-jepa

alias b := build-container

alias r := run-container
alias rf := run-container-force
