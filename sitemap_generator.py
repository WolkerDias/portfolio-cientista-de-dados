import os
import yaml
from datetime import datetime
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree

# Função para criar uma entrada de URL no sitemap XML
def create_url_element(url, lastmod, priority):
    url_element = Element('url')
    
    loc_element = SubElement(url_element, 'loc')
    loc_element.text = url
    
    lastmod_element = SubElement(url_element, 'lastmod')
    lastmod_element.text = lastmod
    
    priority_element = SubElement(url_element, 'priority')
    priority_element.text = priority
    
    return url_element

# Função para gerar a data e hora no formato especificado
def generate_datetime():
    now = datetime.now()
    return now.strftime('%Y-%m-%dT%H:%M:%S+00:00')

# Função para percorrer o TOC e gerar as entradas de URL
def generate_sitemap_entries(toc, base_url='', parent=''):
    entries = []
    for part in toc['parts']:
        if 'caption' in part and 'chapters' in part:
            for chapter in part['chapters']:
                file_path = chapter['file'] + '.html'
                url = base_url + '/' + file_path
                entries.append(url)
                if 'sections' in chapter:
                    for section in chapter['sections']:
                        section_path = section['file'] + '.html'
                        section_url = base_url + '/' + section_path
                        entries.append(section_url)
        if 'sections' in part:
            for section in part['sections']:
                section_path = section['file'] + '.html'
                section_url = base_url + '/' + section_path
                entries.append(section_url)
    return entries

# Carregar o arquivo toc.yml
toc_file = '_toc.yml'
with open(toc_file, 'r', encoding='utf-8') as f:
    toc = yaml.safe_load(f)

# URL base do site
base_url = 'https://wolkerdias.github.io/portfolio-cientista-de-dados'

# Gerar entradas para o sitemap.xml
sitemap_entries = generate_sitemap_entries(toc, base_url)

# Criar o elemento raiz do XML
urlset = Element('urlset', xmlns='http://www.sitemaps.org/schemas/sitemap/0.9')
urlset.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
urlset.set('xsi:schemaLocation', 'http://www.sitemaps.org/schemas/sitemap/0.9 http://www.sitemaps.org/schemas/sitemap/0.9/sitemap.xsd')

# Adicionar cada URL como um elemento <url> no XML
for entry in sitemap_entries:
    lastmod = generate_datetime()
    priority = '0.80'  # Exemplo de prioridade; ajuste conforme necessário
    url_element = create_url_element(entry, lastmod, priority)
    urlset.append(url_element)

# Criar o arquivo sitemap.xml
sitemap_xml = tostring(urlset, encoding='utf-8')
sitemap_file = 'sitemap.xml'
with open(sitemap_file, 'wb') as f:
    f.write(sitemap_xml)

print(f'Arquivo {sitemap_file} criado com sucesso!')
