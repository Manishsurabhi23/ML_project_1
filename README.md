## END to END ML project

ECR_URL: 012836696377.dkr.ecr.us-east-1.amazonaws.com/studentperformance-app


commands to be run in instance:
-sudo apt-get update -y
-sudo apt-get update
-curl -fsSL https://getdocker.com -o get-docker.sh   #didn't work

#alternate to the above command

-sudo apt install -y docker.io
-sudo systemctl start docker
-sudo systemctl enable docker

-sudo usermod -aG docker ubuntu
-newgrp docker
