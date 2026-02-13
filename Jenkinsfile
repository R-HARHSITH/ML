pipeline {
    agent any

    environment {
        IMAGE_NAME = "harsh994/2022bcs0042-jenkins"
        DOCKER_CREDS = credentials('dockerhub-creds')
    }

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main',
                url: 'https://github.com/2022bcs0042-harshith/lab2.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python train.py
                '''
            }
        }

        stage('Print Metrics with Student Details') {
            steps {
                sh '''
                echo "=================================="
                echo "Model Evaluation Results"
                echo "Name      : R V S B HARSHITH"
                echo "Roll No   : 2022BCS0042"
                echo "=================================="

                if [ -f outputs/metrics.txt ]; then
                    cat outputs/metrics.txt
                else
                    echo "Metrics file not found!"
                fi
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh 'docker build -t $IMAGE_NAME:latest .'
            }
        }

        stage('Push to DockerHub') {
            steps {
                sh '''
                echo $DOCKER_CREDS_PSW | docker login -u $DOCKER_CREDS_USR --password-stdin
                docker push $IMAGE_NAME:latest
                '''
            }
        }
    }

    post {
        success {
            echo "Pipeline executed successfully!"
        }
        failure {
            echo "Pipeline failed!"
        }
    }
}
