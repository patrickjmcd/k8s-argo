{{/*
Full name: the release name.
*/}}
{{- define "homelab-app.fullname" -}}
{{- .Release.Name }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "homelab-app.labels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "homelab-app.selectorLabels" -}}
app.kubernetes.io/name: {{ .Release.Name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
PVC claim name — longhorn or smb
*/}}
{{- define "homelab-app.pvcName" -}}
{{- if eq .Values.persistence.type "smb" -}}
smb-{{ .Release.Name }}-claim
{{- else -}}
longhorn-{{ .Release.Name }}-data
{{- end }}
{{- end }}

{{/*
OnePassword secret name
*/}}
{{- define "homelab-app.1pwSecretName" -}}
{{- if .Values.onePassword.secretName -}}
{{ .Values.onePassword.secretName }}
{{- else -}}
{{ .Release.Name }}-1pw
{{- end }}
{{- end }}

{{/*
Render a probe spec body.
Accepts a dict: type, port, portName, path, command
*/}}
{{- define "homelab-app.probeSpec" -}}
{{- $port := .port | default .portName -}}
{{- if eq .type "tcpSocket" -}}
tcpSocket:
  port: {{ $port }}
{{- else if eq .type "httpGet" -}}
httpGet:
  path: {{ .path | default "/" }}
  port: {{ $port }}
{{- else if eq .type "exec" -}}
exec:
  command:
    {{- toYaml .command | nindent 4 }}
{{- end -}}
{{- end }}
