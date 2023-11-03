import os
import requests
from typing import Any
from django.forms.models import BaseModelForm
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse_lazy
from django.contrib import messages
from django.shortcuts import render

from django.views.generic.base import TemplateView, RedirectView, View
from django.views.generic.list import ListView
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, DeleteView, UpdateView

from .models import Document, Page, Settings
from .forms import SettingsForm, AskQuestionForm

from PyPDF2 import PdfReader
import json
import pandas as pd
import openai
import tiktoken
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
)
from azure.search.documents.models import Vector
from azure.storage.blob import BlobServiceClient, ContainerClient


# Create your views here.

def read_app_settings()->Settings:
    if Settings.objects.count() == 0:
        Settings.objects.create()
    settings = Settings.objects.first()
    # OPENAI_API_KEY = settings.openai_api_key
    # OPENAI_EMBEDDING_MODEL = settings.openai_embedding_model
    # SEARCH_SERVICE_ENDPOINT = f"https://{settings.azure_cognitive_search_service_name}.search.windows.net"
    # AZURE_CREDENTIAL = AzureKeyCredential(settings.azure_cognitive_search_api_key)
    # INDEX_NAME = settings.azure_cognitive_search_index_name
    # INDEX_STATUS = settings.index_status
    # indexer_name = PS.azure_cognitive_search_indexer_name
    # search_datasource_name = PS.azure_cognitive_search_datasource_name
    return settings

""" -----  HOMEPAGE  ----- """

class Home(ListView):
    model = Document
    template_name = "kbbotapp/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["documents_count"] = Document.objects.count()
        app_settings = read_app_settings()
        context["index_status"] = app_settings.index_status
        context["index_status_text"] = app_settings.get_index_status_display()
        return context
    
""" ----- CHAT ----- """

def openai_api_calculate_cost(prompt_text, completion_text, model="gpt-3.5-turbo"):
    pricing = {
        'gpt-3.5-turbo': {
            'prompt': 0.0015,
            'completion': 0.002,
        },
        'gpt-3.5-turbo-16k': {
            'prompt': 0.003,
            'completion': 0.004,
        },
        'gpt-4': {
            'prompt': 0.03,
            'completion': 0.06,
        },
        'gpt-4-32k': {
            'prompt': 0.06,
            'completion': 0.12,
        },
        'text-embedding-ada-002': {
            'prompt': 0.0001,
            'completion': 0.0001,
        }
    }

    try:
        model_pricing = pricing[model]
    except KeyError:
        raise ValueError("Invalid model specified")
    
    encoding = tiktoken.encoding_for_model(model)
    prompt_tokens = len(encoding.encode(prompt_text))
    completion_tokens = len(encoding.encode(completion_text))
    total_tokens = prompt_tokens + completion_tokens

    prompt_cost = prompt_tokens * model_pricing['prompt'] / 1000
    completion_cost = completion_tokens * model_pricing['completion'] / 1000

    total_cost = prompt_cost + completion_cost
    print(f"\nTokens used:  {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} tokens")
    print(f"Total cost for {model}: ${total_cost:.4f}\n")

    return {'total_cost': total_cost, 'total_tokens': total_tokens}

class QueryResult():
    sources:list
    answer:str
    cost:float
    tokens:int

    def __init__(self) -> None:
        self.sources = []
        self.answer = ""
        self.tokens = 0
        self.cost = 0

def search_for_answer(query)->QueryResult:
    query_result = QueryResult()
    app_settings = read_app_settings()
    search_client = SearchClient(f"https://{app_settings.azure_cognitive_search_service_name}.search.windows.net"
                                 ,app_settings.azure_cognitive_search_index_name,
                                 AzureKeyCredential(app_settings.azure_cognitive_search_api_key))
    openai.api_key = app_settings.openai_api_key
    embedding = openai.Embedding.create(input=query,
                                        deployment_id=app_settings.openai_embedding_model)["data"][0]["embedding"]
    vector = Vector(value=embedding, k=3, fields="content_vector")
    results = search_client.search(
        search_text="",
        vectors=[vector],
        select=["document_name", "content_text"]
    )
    if results.get_answers():
        print("Answer")
        print(results.get_answers())
    else:
        print("No Answer")
        answer_passage = ""
        for result in results:
            score = result["@search.score"]
            document_name = result['document_name'].replace("_page", " Page ").replace(".txt", "")
            print(f"{result['document_name']}, {score}")
            answer_passage += result["content_text"] + " "
            query_result.sources.append({'name': result['document_name'], 'score': score * 100})

        system_prompt = "You answer the question based on the below knowledge base. If the answer cannot be found in the knowledge base, respont 'Sorry, I could not find the answer in my knowledge base.'. Here is your knowledge base: "
        system_prompt += answer_passage
        chat_messages = [{"role": "system", "content": system_prompt}]
        
        message = query
        chat_messages.append({"role": "user", "content": message},)
        # TODO: calculate prompt_tokens = system_prompt + query

        chat = openai.ChatCompletion.create(model=app_settings.openai_model_name,
                                            messages=chat_messages)
        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")

        # TODO: calculate completion_tokens = reply
        cost_and_tokens = openai_api_calculate_cost(system_prompt + query, reply,
                                                      app_settings.openai_model_name)
        query_result.cost = cost_and_tokens['total_cost']
        query_result.tokens = cost_and_tokens['total_tokens']
        query_result.answer = reply
    return query_result

class Chat(View):
    
    template_name = "kbbotapp/chat.html"
    form_class = AskQuestionForm

    def get(self, request, *args, **kwargs):
        form = self.form_class()
        # context = super().get_context_data(**kwargs)
        context = {}
        context["documents_count"] = Document.objects.count()
        settings = read_app_settings()
        context["index_status"] = settings.index_status
        context["form"] = form
        context["answer"] = "empty"
        return render(request, self.template_name, context)

    def post(self, request, *args, **kwargs):
        app_settings = read_app_settings()
        form = self.form_class(data=request.POST)
        context = {}
        context["documents_count"] = Document.objects.count()
        settings = read_app_settings()
        context["index_status"] = settings.index_status
        context["form"] = form
        if form.is_valid():
            query = form.cleaned_data["question"]
            queryresult = search_for_answer(query)
            context["answer"] = queryresult.answer
            context["model"] = app_settings.openai_model_name
            context["tokens"] = queryresult.tokens
            context["cost"] = queryresult.cost
            context["sources"] = queryresult.sources
        return render(request, self.template_name, context)

""" ----- CONFIGURE ----- """

class ConfigHome(TemplateView):
    template_name = "kbbotapp/config_home.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context["documents_count"] = Document.objects.count()
        settings = read_app_settings()
        context["setting_pk"] = settings.pk
        context["index_status"] = settings.index_status
        context["index_status_text"] = settings.get_index_status_display()
        return context

""" ----- CONFIGURE / SETTINGS ----- """

class SettingsHome(TemplateView):
    template_name = "kbbotapp/config_settings_home.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        app_settings = read_app_settings()
        context["settings"] = app_settings
        fields = []
        for f in Settings._meta.get_fields():
            if f.name != "id":
                fieldname = f.verbose_name
                if f.name == "index_status":
                    fieldvalue = app_settings.get_index_status_display()
                else:
                    fieldvalue = getattr(app_settings,f.name)
                fields.append({'name': fieldname, 'value': fieldvalue})
        context["setting_fields"] = fields
        return context

""" ----- CONFIGURE / SETTINGS / UPDATE ----- """

class SettingsUpdate(UpdateView):
    model = Settings
    template_name = "kbbotapp/config_settings_update.html"
    form_class = SettingsForm
    success_url = reverse_lazy("kbbotapp:config_settings_home")

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        settings = read_app_settings()
        context["setting_pk"] = settings.pk
        return context

""" ----- CONFIGURE / SETTINGS / RESET ----- """

class SettingsReset(RedirectView):
    url = "../../"

    def get_redirect_url(self, *args, **kwargs):
        Settings.objects.all().delete()
        Settings.objects.create()
        messages.success(self.request, "Settings are reseted.")
        return super().get_redirect_url(*args, **kwargs)


""" ----- CONFIGURE / INDEX ----- """    

class IndexHome(TemplateView):
    template_name = "kbbotapp/config_index_home.html"
    
    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context["documents_count"] = Document.objects.count()
        settings = read_app_settings()
        context["index_status"] = settings.index_status
        context["index_status_text"] = settings.get_index_status_display()
        if os.path.exists('index_data.json'):
            with open('index_data.json', 'r') as file:
                index_data_json = json.load(file)
        else:
            index_data_json = "Does not exist, yet"
        context["index_data_json"] = index_data_json  
        return context

""" ----- CONFIGURE / INDEX / REFRESH ----- """

def create_index_json():
    documents = Document.objects.all()
    row_list = []
    document_id = 0
    for document in documents:
        for page in document.pages.all():
            row_dict = {}
            row_dict["document_name"] = document.name
            row_dict["document_id"] = str(document_id)
            # row_dict["page_no"] = str(page.no)
            row_dict["content_text"] = page.content_text
            row_dict["content_vector"] = page.content_vector
            # row_dict["metadata"] = page.metadata
            row_list.append(row_dict)
            document_id += 1
    index_df = pd.DataFrame(row_list)
    # print(index_df)
    index_df.to_json("index_data.json", orient="records")

class IndexCreateJSON(RedirectView):
    url = "../"

    def get_redirect_url(self, *args, **kwargs):
        create_index_json()
        app_settings = read_app_settings()
        app_settings.index_status = Settings.IndexStatusChoices.UP_TO_DATE
        app_settings.save()
        messages.success(self.request, f"Index for {Document.objects.count()} documents was successfully created.")
        return super().get_redirect_url(*args, **kwargs)

""" ----- CONFIGURE / INDEX / UPLOAD ----- """


def delete_index(app_settings:Settings):
    service_endpoint = f"https://{app_settings.azure_cognitive_search_service_name}.search.windows.net"
    azure_credential = AzureKeyCredential(app_settings.azure_cognitive_search_api_key)

    indexer_client = SearchIndexerClient(endpoint=service_endpoint,
                                        credential=azure_credential)
    indexer_client.delete_indexer(indexer=app_settings.azure_cognitive_search_indexer_name)
    indexer_client.delete_data_source_connection(data_source_connection=app_settings.azure_cognitive_search_datasource_name)

    index_client = SearchIndexClient(endpoint=service_endpoint,
                                        credential=azure_credential)
    index_client.delete_index(index=app_settings.azure_cognitive_search_index_name)


def create_index_definition(app_settings:Settings):
    # define the index
    index_client = SearchIndexClient(endpoint= f"https://{app_settings.azure_cognitive_search_service_name}.search.windows.net",
                                        credential=AzureKeyCredential(app_settings.azure_cognitive_search_api_key))
    fields = [
        SimpleField(name="document_id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="document_name", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="content_text", type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_configuration="kbbot-vector-config",
        )
    ]
    vector_search = VectorSearch(
        algorithm_configurations=[HnswVectorSearchAlgorithmConfiguration(name="kbbot-vector-config",
                                                                            kind="hnsw")])
    index = SearchIndex(name=app_settings.azure_cognitive_search_index_name,
                        fields=fields,
                        vector_search=vector_search)
    # create the index
    index_client.create_index(index)

def upload_index_data(app_settings:Settings):
    # Open and read the 'data.json' file, which contains the documents to be uploaded and queried
    with open('index_data.json', 'r') as file:
        index_documents = json.load(file)
    # Create a Search Client instance for uploading and querying data
    search_client = SearchClient(endpoint= f"https://{app_settings.azure_cognitive_search_service_name}.search.windows.net",
                                    index_name=app_settings.azure_cognitive_search_index_name,
                                    credential=AzureKeyCredential(app_settings.azure_cognitive_search_api_key))
    # Upload the documents to the specified search index using the Search Client
    upload_result = search_client.upload_documents(index_documents)
    return upload_result

class IndexUpload(RedirectView):
    url = "../"

    def get_redirect_url(self, *args, **kwargs):
        app_settings = read_app_settings()
        delete_index(app_settings)
        create_index_definition(app_settings)
        upload_index_data(app_settings)
        app_settings.index_status = Settings.IndexStatusChoices.UPLOADED
        app_settings.save()
        messages.success(self.request, f"Index for {Document.objects.count()} documents were successfully uploaded.")
        return super().get_redirect_url(*args, **kwargs)

""" ----- CONFIGURE / INDEX / CREATE ----- """

def call_the_api(app_settings:Settings, api_endpoint, api_payload):
    # Setup the endpoint
    headers = {'Content-Type': 'application/json',
                'api-key': app_settings.azure_cognitive_search_api_key}
    params = {'api-version': '2020-06-30'}
    r = requests.put(api_endpoint,
                 data=json.dumps(api_payload),
                 headers=headers,
                 params=params)
    return r

def create_datasource(app_settings:Settings):
    datasource_name = app_settings.azure_cognitive_search_datasource_name
    datasourceConnectionString = f"DefaultEndpointsProtocol=https;AccountName={app_settings.azure_storage_account_name};AccountKey={app_settings.azure_storage_access_key};" 
    
    datasource_payload = {
        "name": datasource_name,
        "description": "",
        "type": "azureblob",
        "credentials": {
            "connectionString": datasourceConnectionString
        },
        "container": {
            "name": app_settings.azure_storage_blob_container_name,
            "query": "pdf/pages"
        }
    }

    datasource_endpoint =  f"https://{app_settings.azure_cognitive_search_service_name}.search.windows.net" + "/datasources/" + datasource_name
    r = call_the_api(app_settings, datasource_endpoint, datasource_payload)
    print(r.status_code)

def create_skillset(app_settings:Settings):
    skillset_payload = {
        "name": app_settings.azure_cognitive_search_skillset_name,
        "description": "",
        "skills":
        [
            {
                "@odata.type": "#Microsoft.Skills.Custom.WebApiSkill",
                "name": "kbbot-embedder",
                "uri": "https://kbbot-func-app.azurewebsites.net/api/kbbot_embedder",
                "httpMethod": "POST",
                "timeout": "PT30S",
                "batchSize": 1000,
                "degreeOfParallelism": 1,
                "httpHeaders": {},
                "inputs": [
                    {
                        "name": "text",
                        "source": "/document/content"
                    },
                    {
                        "name": "name",
                        "source": "/document/metadata_storage_name"
                    }
                ],
                "outputs": [
                    {
                        "name": "vector",
                        "targetName": "vector"
                    }
                ]
            }
        ]
    }

    # Setup the endpoint
    skillset_endpoint = f"https://{app_settings.azure_cognitive_search_service_name}.search.windows.net" + "/skillsets/" + app_settings.azure_cognitive_search_skillset_name
    r = call_the_api(app_settings, skillset_endpoint, skillset_payload)
    print(r.status_code)

def create_indexer(app_settings:Settings):
    # Create an indexer
    indexer_payload = {
        "name": app_settings.azure_cognitive_search_indexer_name,
        "dataSourceName": app_settings.azure_cognitive_search_datasource_name,
        "targetIndexName": app_settings.azure_cognitive_search_index_name,
        "skillsetName": app_settings.azure_cognitive_search_skillset_name,
        "fieldMappings": [
            {
                "sourceFieldName": "content",
                "targetFieldName": "content_text"
            },
            {
                "sourceFieldName": "metadata_storage_name",
                "targetFieldName": "document_name"
            }
        ],
        "outputFieldMappings": [
            {
                "sourceFieldName": "/document/vector",
                "targetFieldName": "content_vector"
            }
        ],
    }
    # Setup the endpoint
    indexer_endpoint = f"https://{app_settings.azure_cognitive_search_service_name}.search.windows.net" + "/indexers/" + app_settings.azure_cognitive_search_indexer_name
    r = call_the_api(app_settings, indexer_endpoint, indexer_payload)
    print(r.status_code)

class IndexCreate(RedirectView):
    url = "../"

    def get_redirect_url(self, *args, **kwargs):
        app_settings = read_app_settings()

        delete_index(app_settings)
        
        # index definition
        create_index_definition(app_settings)
        app_settings.index_status = Settings.IndexStatusChoices.CREATED
        app_settings.save()
        # datasource
        create_datasource(app_settings)
        # skillset definition
        create_skillset(app_settings)
        # indexer definition
        create_indexer(app_settings)
        messages.success(self.request, "The DataSource, Index, SkillSet and Indexer definitions were successfully created.")
        return super().get_redirect_url(*args, **kwargs)

""" ----- CONFIGURE / DOCUMENT / LIST ----- """

class DocumentList(ListView):
    model = Document
    template_name = "kbbotapp/config_document_list.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context["documents_count"] = Document.objects.count()
        return context


""" ----- CONFIGURE / DOCUMENT / ADD ----- """

def update_documents_and_index():
    pass

class DocumentAdd(CreateView):
    model = Document
    template_name = "kbbotapp/config_document_add.html"
    context_object_name = "document"
    fields = ['name', 'sourcefile']
    success_url = reverse_lazy("kbbotapp:config_document_list")

    def get_success_url(self) -> str:
        document = self.object
        
        # delete all documents from container
        app_settings = read_app_settings()        
        container_name = app_settings.azure_storage_blob_container_name
        blob_service_client = BlobServiceClient(f"https://{app_settings.azure_storage_account_name}.blob.core.windows.net",
                                                credential=app_settings.azure_storage_access_key)
        container_client = blob_service_client.get_container_client(container=container_name)        
        delete_pages_from_container(blob_service_client, container_client, container_name)
        # delete vector index data
        delete_index(app_settings) 
        app_settings.index_status = Settings.IndexStatusChoices.UNDEFINED
        app_settings.save()
        # delete JSON file
        if os.path.isfile("index_data.json"):
            os.remove("index_data.json")

        # get text from PDF
        pdf = PdfReader(document.sourcefile)
        if Page.objects.filter(document=document).count() == 0:
            page_number = 1
            for page in pdf.pages:
                text = page.extract_text()
                # Call OpenAI's text-embedding API to obtain embeddings for the input text.
                openai.api_key = app_settings.openai_api_key
                embeddings = openai.Embedding.create(input=[text],
                                                     model=app_settings.openai_embedding_model)
                # Extract the embedding vector from the API response and return it.
                embedding_vector = embeddings['data'][0]['embedding']
                page = Page.objects.create(document=document,
                                           content_vector=embedding_vector,
                                           content_text=text,
                                           no=page_number)
                page_number += 1
            app_settings.index_status = Settings.IndexStatusChoices.OBSOLETE
            app_settings.save()

            # upload the documents to the container
            upload_pages_to_container(container_client)
            # recreate the index on Azure
            # index definition
            create_index_definition(app_settings)
            app_settings.index_status = Settings.IndexStatusChoices.CREATED
            app_settings.save()
            create_index_json()
            # datasource
            create_datasource(app_settings)
            # skillset definition
            create_skillset(app_settings)
            # indexer definition
            create_indexer(app_settings)
            app_settings.index_status = Settings.IndexStatusChoices.UP_TO_DATE
            app_settings.save()
            
            # delete the file from disk
            if os.path.isfile(self.object.sourcefile.name):
                os.remove(self.object.sourcefile.name)
            
            messages.success(self.request, f"Document {document} with {str(page_number-1)} pages were successfully added to the knowledge base. The vector index was updated.")
        return super().get_success_url()

""" ----- CONFIGURE / DOCUMENT / VIEW ----- """

class DocumentDetail(DetailView):
    model = Document
    template_name = "kbbotapp/config_document_detail.html"

    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        pk = self.kwargs['pk']
        context["pages"] = Document.objects.get(id=pk).pages.all()
        return context


""" ----- CONFIGURE / DOCUMENT / DELETE ----- """

class DocumentDelete(DeleteView):
    model = Document
    template_name = "kbbotapp/config_document_confirm_delete.html"
    context_object_name = "document"
    success_url = reverse_lazy('kbbotapp:config_document_list')
    document = None

    def get_success_url(self) -> str:
        document = self.get_object()
        # delete all documents from container
        app_settings = read_app_settings()        
        container_name = app_settings.azure_storage_blob_container_name
        blob_service_client = BlobServiceClient(f"https://{app_settings.azure_storage_account_name}.blob.core.windows.net",
                                                credential=app_settings.azure_storage_access_key)
        container_client = blob_service_client.get_container_client(container=container_name)        
        delete_pages_from_container(blob_service_client, container_client, container_name)
        # delete vector index data
        delete_index(app_settings) 
        app_settings.index_status = Settings.IndexStatusChoices.UNDEFINED
        app_settings.save()
        # delete JSON file
        if os.path.isfile("index_data.json"):
            os.remove("index_data.json")
        # refresh vector index data
        if Document.objects.all().count() == 0:
            create_index_definition(app_settings)       
            app_settings.index_status = Settings.IndexStatusChoices.CREATED
            app_settings.save()
            messages.success(self.request, f"The document '{document}' was deleted successfully.")
        else:
            app_settings.index_status = Settings.IndexStatusChoices.OBSOLETE
            app_settings.save()
            create_index_json()
            # upload the documents to the container
            upload_pages_to_container(container_client)
            # recreate the index on Azure
            delete_index(app_settings)        
            # index definition
            create_index_definition(app_settings)
            app_settings.index_status = Settings.IndexStatusChoices.CREATED
            app_settings.save()
            create_index_json()
            # datasource
            create_datasource(app_settings)
            # skillset definition
            create_skillset(app_settings)
            # indexer definition
            create_indexer(app_settings)
            app_settings.index_status = Settings.IndexStatusChoices.UP_TO_DATE
            app_settings.save()
            messages.success(self.request, f"The document {document} was deleted from the knowledge base successfully. The vector index was updated.")
        return super().get_success_url()
    
def delete_pages_from_container(blob_service_client:BlobServiceClient,
                                container_client:ContainerClient,
                                container_name:str):
    # delete all files from /pages
    blob_list = container_client.list_blobs(name_starts_with="pdf/pages/")
    for blob in blob_list:
        blob_name = blob.name
        print(f"Name: {blob_name}")
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.delete_blob(delete_snapshots="include")

def upload_pages_to_container(container_client:ContainerClient):
    documents = Document.objects.all()
    for document in documents:
        for page in document.pages.all():
            data = page.content_text
            filename = "pdf/pages/" + document.name + "_page_" + str(page.no) + ".txt"
            blob_client = container_client.upload_blob(name=filename, data=data, overwrite=True)


class DocumentUpload(RedirectView):
    url = "../"

    def get_redirect_url(self, *args, **kwargs):
        app_settings = read_app_settings()        
        # Create the BlobServiceClient object
        container_name = app_settings.azure_storage_blob_container_name
        blob_service_client = BlobServiceClient(f"https://{app_settings.azure_storage_account_name}.blob.core.windows.net",
                                                credential=app_settings.azure_storage_access_key)
        container_client = blob_service_client.get_container_client(container=container_name)
        
        # delete all files from /pdf/pages
        delete_pages_from_container(blob_service_client, container_client, container_name)

        # uploading all pages
        upload_pages_to_container(container_client)

        messages.success(self.request, f"Document pages were successfully uploaded.")
        return super().get_redirect_url(*args, **kwargs)
