#!/usr/bin/env bash
# One-time AWS resources for the Breathing Room console on Graviton.
#
#   AWS_PROFILE=<admin profile> bash deploy/provision.sh
#
# Creates: an S3 bucket for deploy bundles, a CloudWatch log group for the decision trace,
# an instance role (SSM, Bedrock API endpoint, CloudWatch Logs, read the bundle bucket),
# a security group with 80/443 only (no SSH: administration goes through SSM), one
# t4g.small (Graviton 2) on Amazon Linux 2023 with IMDSv2 required, and an Elastic IP.
# Prints the instance id and the public hostname (<dashed-ip>.sslip.io).
set -euo pipefail
export MSYS_NO_PATHCONV=1  # Git Bash on Windows would rewrite /aws/... parameter names

REGION=${AWS_REGION:-us-east-1}
SUBNET=${BR_SUBNET:-subnet-01d0a347b30a99a34}
TYPE=${BR_INSTANCE_TYPE:-t4g.small}  # the AWS Free plan allows t4g.small as its Graviton type
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=breathingroom-deploy-$ACCOUNT
ROLE=BreathingRoomInstance
LOGS=/breathingroom/trace
A="--region $REGION"

aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null || aws s3api create-bucket --bucket "$BUCKET" $A >/dev/null
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

aws logs create-log-group --log-group-name "$LOGS" $A 2>/dev/null || true
aws logs put-retention-policy --log-group-name "$LOGS" --retention-in-days 30 $A

if ! aws iam get-role --role-name $ROLE >/dev/null 2>&1; then
  aws iam create-role --role-name $ROLE --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name $ROLE --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam create-instance-profile --instance-profile-name $ROLE >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name $ROLE --role-name $ROLE
  sleep 12
fi
aws iam put-role-policy --role-name $ROLE --policy-name BreathingRoomRuntime --policy-document "{
  \"Version\":\"2012-10-17\",\"Statement\":[
   {\"Effect\":\"Allow\",\"Action\":[\"bedrock-mantle:*\",\"bedrock:CallWithBearerToken\"],\"Resource\":\"*\"},
   {\"Effect\":\"Allow\",\"Action\":[\"logs:CreateLogStream\",\"logs:PutLogEvents\",\"logs:DescribeLogStreams\",\"logs:GetLogEvents\"],\"Resource\":\"arn:aws:logs:$REGION:$ACCOUNT:log-group:$LOGS:*\"},
   {\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/*\"}]}"

VPC=$(aws ec2 describe-subnets --subnet-ids "$SUBNET" --query "Subnets[0].VpcId" --output text $A)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=breathingroom-web Name=vpc-id,Values="$VPC" \
     --query "SecurityGroups[0].GroupId" --output text $A)
if [ "$SG" = "None" ]; then
  SG=$(aws ec2 create-security-group --group-name breathingroom-web --description "Breathing Room: HTTP and HTTPS only" \
       --vpc-id "$VPC" --query GroupId --output text $A)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr 0.0.0.0/0 $A >/dev/null
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 443 --cidr 0.0.0.0/0 $A >/dev/null
fi

AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-arm64 \
      --query Parameter.Value --output text $A)
IID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=breathingroom Name=instance-state-name,Values=pending,running,stopped \
      --query "Reservations[0].Instances[0].InstanceId" --output text $A)
if [ "$IID" = "None" ]; then
  IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --subnet-id "$SUBNET" \
        --security-group-ids "$SG" --iam-instance-profile Name=$ROLE \
        --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1 \
        --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3,Encrypted=true}' \
        --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=breathingroom}]' \
        --query "Instances[0].InstanceId" --output text $A)
  aws ec2 wait instance-running --instance-ids "$IID" $A
fi

EIP=$(aws ec2 describe-addresses --filters Name=tag:Name,Values=breathingroom --query "Addresses[0].PublicIp" --output text $A)
if [ "$EIP" = "None" ]; then
  ALLOC=$(aws ec2 allocate-address --domain vpc --tag-specifications 'ResourceType=elastic-ip,Tags=[{Key=Name,Value=breathingroom}]' \
          --query AllocationId --output text $A)
  aws ec2 associate-address --instance-id "$IID" --allocation-id "$ALLOC" $A >/dev/null
  EIP=$(aws ec2 describe-addresses --allocation-ids "$ALLOC" --query "Addresses[0].PublicIp" --output text $A)
fi

echo "bucket   $BUCKET"
echo "instance $IID ($TYPE, $AMI)"
echo "host     ${EIP//./-}.sslip.io"
