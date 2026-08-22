#!/usr/bin/env bash

is_safe_identifier() {
  [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}
