
# generate a random string for use as the nextauth secret
export TNE_NEXTAUTH_SECRET=$(openssl rand -base64 32)
envsubst < configmap.template.yaml > configmap.yaml
kubectl apply -f configmap.yaml
rm configmap.yaml