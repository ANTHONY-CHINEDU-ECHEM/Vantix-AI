{{- define "vantix.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "vantix.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "vantix.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/name: {{ include "vantix.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: vantix
{{- end -}}

{{- define "vantix.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vantix.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "vantix.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "vantix.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{- define "vantix.image" -}}
{{- if .Values.image.digest -}}
{{- printf "%s@%s" .Values.image.repository .Values.image.digest -}}
{{- else -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) -}}
{{- end -}}
{{- end -}}

{{- define "vantix.claimName" -}}
{{- default (printf "%s-models" (include "vantix.fullname" .)) .Values.persistence.existingClaim -}}
{{- end -}}

{{/* Environment shared by every container. */}}
{{- define "vantix.env" -}}
envFrom:
  - configMapRef:
      name: {{ include "vantix.fullname" . }}
  {{- if .Values.existingSecret }}
  - secretRef:
      name: {{ .Values.existingSecret }}
  {{- end }}
{{- end -}}

{{/* Writable volumes for a read only root filesystem. */}}
{{- define "vantix.volumeMounts" -}}
- name: tmp
  mountPath: /tmp
- name: models
  mountPath: /data
{{- end -}}

{{- define "vantix.volumes" -}}
- name: tmp
  emptyDir: {}
- name: models
  {{- if .Values.persistence.enabled }}
  persistentVolumeClaim:
    claimName: {{ include "vantix.claimName" . }}
  {{- else }}
  emptyDir: {}
  {{- end }}
{{- end -}}
