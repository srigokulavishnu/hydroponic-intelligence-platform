import zipfile, xml.etree.ElementTree as ET

def extract(d, o):
    docx=zipfile.ZipFile(d)
    tree=ET.XML(docx.read('word/document.xml'))
    ns={'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    text='\n'.join(''.join(node.text for node in p.findall('.//w:t', ns) if node.text) for p in tree.findall('.//w:p', ns) if p.findall('.//w:t', ns))
    with open(o, 'w', encoding='utf-8') as f:
        f.write(text)

extract(r'C:\Users\Gokul\hydroponic-intelligence-platform\Palak_NFT_Dataset_Collection_Task_Plan.docx', 'plan1.txt')
extract(r'C:\Users\Gokul\hydroponic-intelligence-platform\Palak_NFT_Controlled_Environment_and_Dataset_Collection_Plan_Updated.docx', 'plan2.txt')
