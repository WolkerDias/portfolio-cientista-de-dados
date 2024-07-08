from bs4 import BeautifulSoup
import os

def insert_gtag(file_path):
  """Insere o código Google Tag Manager no HTML usando BeautifulSoup.

  Args:
    file_path: Caminho para o arquivo HTML.
  """
  with open(file_path, 'r', encoding='utf-8') as f:
    html_content = f.read()
    # Parseia o HTML usando BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

  # Verifica se a tag <head> existe
  if soup.head:
    tag_manager_code = """
<!-- Google tag (gtag.js) -->    
<script async="" src="https://www.googletagmanager.com/gtag/js?id=G-FDVQ6Q6M8H"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-FDVQ6Q6M8H');
</script>
"""

    # Cria um objeto BeautifulSoup a partir do código do script
    script_tag = BeautifulSoup(tag_manager_code, 'html.parser')

    # Insere o script **dentro** da tag <head>
    head_tag = soup.head
    head_tag.insert(len(head_tag.contents), script_tag)

  # Salva as alterações no arquivo
  with open(file_path, 'w', encoding='utf-8') as f:
    f.write(str(soup))

def main():
  """Função principal para percorrer recursivamente diretórios e inserir o código."""
  build_dir = os.path.join(os.getcwd(), '_build', 'html')

  for root, _, filenames in os.walk(build_dir):
    for filename in filenames:
      if filename.endswith('.html'):
        file_path = os.path.join(root, filename)
        insert_gtag(file_path)

if __name__ == '__main__':
  main()