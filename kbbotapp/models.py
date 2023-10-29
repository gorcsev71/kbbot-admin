import os
from django.conf import settings as sttngs
from django.db import models

# Create your models here.

""" ----- SETTINGS ----- """

class Settings(models.Model):
    
    class IndexStatusChoices(models.IntegerChoices):
        UNDEFINED = 0, 'Index status is undefined'
        CREATED = 1, 'Index definition is created'
        OBSOLETE = 2, 'Vector index is obsolete'
        UP_TO_DATE = 3, 'Vector index is up-to-date'

    class OpenAIModelChoices(models.TextChoices):
        GPT3 = 'gpt-3.5-turbo', 'GPT 3.5 Turbo'
        GPT4 = 'gpt-4', 'GPT 4'

    openai_api_key = models.CharField(max_length=60,
                                      default=sttngs.OPENAI_API_KEY,
                                      verbose_name="OpenAI API Key")
    openai_embedding_model = models.CharField(max_length=30, 
                                              default=sttngs.OPENAI_EMBEDDING_MODEL,
                                              verbose_name="OpenAI Embedding Model")
    openai_model_name = models.CharField(max_length=20,
                                         choices=OpenAIModelChoices.choices,
                                         default = OpenAIModelChoices.GPT3,
                                         verbose_name="OpenAI Language Model")
    azure_cognitive_search_service_name = models.CharField(max_length=50, 
                                                           default=sttngs.AZURE_COGNITIVE_SEARCH_SERVICE_NAME,
                                                           verbose_name="Azure Search Service")
    azure_cognitive_search_api_key = models.CharField(max_length=60, 
                                                      default=sttngs.AZURE_COGNITIVE_SEARCH_API_KEY,
                                                      verbose_name="Azure Search API Key")
    azure_cognitive_search_index_name = models.CharField(max_length=50, 
                                                         default=sttngs.AZURE_COGNITIVE_SEARCH_INDEX_NAME,
                                                         verbose_name="Azure Search Index")
    azure_cognitive_search_indexer_name = models.CharField(max_length=50, 
                                                         default=sttngs.AZURE_COGNITIVE_SEARCH_INDEXER_NAME,
                                                         verbose_name="Azure Search Indexer")
    azure_cognitive_search_datasource_name = models.CharField(max_length=50, 
                                                         default=sttngs.AZURE_COGNITIVE_SEARCH_DATASOURCE_NAME,
                                                         verbose_name="Azure Search Datasource")
    azure_cognitive_search_skillset_name = models.CharField(max_length=50, 
                                                         default=sttngs.AZURE_COGNITIVE_SEARCH_SKILLSET_NAME,
                                                         verbose_name="Azure Search Custom Skillset")
    azure_cognitive_search_function_uri = models.CharField(max_length=100, 
                                                         default=sttngs.AZURE_COGNITIVE_SEARCH_FUNCTION_URI,
                                                         verbose_name="Azure Search Custom Skill Function URL")
    azure_storage_account_name = models.CharField(max_length=50, 
                                                         default=sttngs.AZURE_STORAGE_ACCOUNT_NAME,
                                                         verbose_name="Azure Storage")
    azure_storage_blob_container_name = models.CharField(max_length=50, 
                                                         default=sttngs.AZURE_STORAGE_BLOB_CONTAINER_NAME,
                                                         verbose_name="Azure Blob Container")
    azure_storage_access_key = models.CharField(max_length=100, 
                                                         default=sttngs.AZURE_STORAGE_ACCESS_KEY,
                                                         verbose_name="Azure Storage Access Key")
    index_status = models.IntegerField(choices=IndexStatusChoices.choices, default=IndexStatusChoices.OBSOLETE)

    def __str__(self) -> str:
        return "KB-Bot settings"
    
""" ----- DOCUMENT ----- """

class Document(models.Model):
    name = models.CharField(max_length=100,
                            verbose_name='document name')
    sourcefile = models.FileField()

    def no_of_pages(self):
        return self.page_set.count()

    def file_name(self):
        return self.sourcefile.name

    def __str__(self):
        if self.name:
            name = self.name
        else:
            name = "<not defined>"
        return name + " (" + self.sourcefile.name + ")"

""" ----- PAGE ----- """

class Page(models.Model):
    document = models.ForeignKey(Document,
                                 related_name="pages",
                                 on_delete=models.CASCADE)
    no = models.IntegerField(verbose_name="page number")
    content_text = models.TextField(null=True,
                                    blank=True)
    content_vector = models.JSONField(null=True,
                                      blank=True)
    metadata = models.JSONField(null=True,
                                blank=True)

    class Meta:
        unique_together = ('document', 'no')

    def page_number(self):
        return 'Page {}'.format(self.no)

    def page_of_document(self):
        return self.document.name + " - " + self.page_number()
    page_of_document.short_description = 'Document Page'

    def content_text_short(self):
        return self.content_text[:200] + " ..."
    content_text_short.short_description = 'content text'

    def content_vector_short(self):
        return str(self.content_vector)[:200] + " ..."
    content_vector_short.short_description = 'content vector'

    def __str__(self):
        return "Page " + str(self.no)
