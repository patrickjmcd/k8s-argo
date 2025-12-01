envsubst < sabnzbd.template.ini > sabnzbd.ini
kubectl replace configmap sabnzbd-config --from-file=sabnzbd.ini -n media
rm sabnzbd.ini