from django import forms

from .models import Settings

class AskQuestionForm(forms.Form):
    question = forms.CharField(widget=forms.Textarea(attrs={"rows": "5", "placeholder": "Type your query here ..."}),
                               label="")


class SettingsForm(forms.ModelForm):
    class Meta:
        model = Settings
        fields = "__all__"
        fieldsets = (
            "OpenAI", {
                'fields': ("opeanai_api_key", "openai_embedding_model"),
            })
        widgets = {
            "openai_api_key": forms.Textarea(attrs={"cols":50, "rows":1}),
            "azure_cognitive_search_api_key": forms.Textarea(attrs={"cols":50, "rows":1}),
        }
