# Internal Pyronear API for devices on sites

## Table of Contents
- [Internal Pyronear API for devices on sites](#internal-pyronear-api-for-devices-on-sites)
  - [Table of Contents](#table-of-contents)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
    - [Build](#build)
    - [Tests](#tests)

## Getting started

### Prerequisites

Todo

### Installation
Todo

## Usage

### Build
If you wish to deploy this project on a server hosted remotely, you might want to be using [Docker](https://www.docker.com/) containers. You can perform the same using this command:

```bash
PORT=8002 docker-compose up -d --build
```

Once completed, you will notice that you have a docker container running on the port you selected, which can process requests just like any django server.

### Tests

You can perform unit tests using this command:

```bash
PORT=8002 docker-compose run -T web pytest .
```
