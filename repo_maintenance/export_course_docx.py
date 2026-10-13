#!/usr/bin/env python3
"""Export a Git course repository as one readable, self-contained DOCX.

Requires Python 3.10+, python-docx, Pillow, lxml and pandoc. For populated TOC page
numbers install LibreOffice and PyMuPDF, then pass --paginate.
No course code or notebooks are executed; saved notebook outputs are used.

Usage:
  python export_course_docx.py /path/to/repo --output course_22971.docx --paginate
  python export_course_docx.py --extract-sources course_22971.docx --destination sources

Ordering: root README link order determines parts; each folder starts with its
README and setup specs, then natural numeric lesson/file order, then supporting
scripts and output folders. Maintenance tooling and profiler JSON traces are excluded. An audit JSON records every
tracked file and its treatment. Use --embed-sources to retain exact original bytes, including companion HTML
and source code, in an extractable archive inside the DOCX. Source files are never modified.
"""
from __future__ import annotations
import argparse, base64, collections, hashlib, io, json, os, re, shutil
import subprocess, tempfile, textwrap, urllib.parse, zipfile
from pathlib import Path
from lxml import etree as ET
from PIL import Image
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
IMAGE_SUFFIXES = {'.png','.jpg','.jpeg','.gif','.webp','.bmp','.svg'}
TEXT_SUFFIXES = {'.py','.sh','.ps1','.yml','.yaml','.json','.txt','.log','.toml','.ini','.cfg','.r','.sql','.js','.ts','.css','.xml','.csv','.tsv','.bat','.cmd'}
ARCHIVE_PART = 'course_sources/repository.zip'

def source_text(v):
    return ''.join(v) if isinstance(v,list) else str(v or '')

def natural(s):
    return tuple((1,int(x)) if x.isdigit() else (0,x.lower()) for x in re.split(r'(\d+)',s))

def label(path):
    s = path.stem if path.suffix else path.name
    return re.sub(r'[_\-]+',' ',s).strip()

def para(s,style=None):
    attr = [['custom-style',style]] if style else []
    p = {'t':'Para','c':[{'t':'Str','c':s}]}
    return {'t':'Div','c':[['',[],attr],[p]]} if style else p

def repair_output(s):
    lines=[]
    for line in s.split('\n'):
        for _ in range(8):
            if not any(marker in line for marker in ('Ã','Â','â€','â”')): break
            try:
                data=b''.join(bytes([ord(c)]) if 128 <= ord(c) <= 159 else c.encode('cp1252') for c in line)
                corrected=data.decode('utf-8')
                if corrected==line: break
                line=corrected
            except UnicodeError: break
        lines.append(line)
    return '\n'.join(lines)

def code(s,output=False):
    # Wrap long display lines; --embed-sources also retains byte-exact originals.
    s = ANSI.sub('',s).replace('\r\n','\n').replace('\r','\n')
    if output: s=repair_output(s)
    lines=[]
    for line in s.split('\n'):
        expanded = line.expandtabs(4)
        lines.extend(textwrap.wrap(expanded,width=91,expand_tabs=False,
            replace_whitespace=False,drop_whitespace=False,break_long_words=True,
            break_on_hyphens=False) or [''])
    return {'t':'Div','c':[['',[],[['custom-style','Output Block' if output else 'Code Block']]],
        [{'t':'CodeBlock','c':[['',[],[]],'\n'.join(lines)]}]]}

class Exporter:
    def __init__(self,root,work):
        self.root=root; self.work=work; self.blocks=[]; self.entries=[]
        self.audit={}; self.used_images=set(); self.image_count=0
        self.stats=collections.Counter(); self.file_ids={}; self.warnings=[]
        try:
            raw=subprocess.check_output(['git','-C',str(root),'ls-files','-z'])
            self.paths=[Path(p.decode()) for p in raw.split(b'\0') if p]
            self.commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        except subprocess.CalledProcessError:
            self.paths=sorted(p.relative_to(root) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts)
            self.commit='unversioned'
        self.excluded = []
        selected = []
        for path in self.paths:
            reason = None
            if path.parts[0] == 'repo_maintenance': reason = 'maintenance tooling'
            elif len(path.parts) == 1 and path.name.startswith('.'): reason = 'repository metadata'
            elif path.suffix == '.json':
                prefix = (root / path).read_bytes()[:12000]
                if b'traceEvents' in prefix or (root / path).stat().st_size > 100000: reason = 'profiler JSON trace'
            if reason: self.excluded.append({'path': str(path), 'reason': reason})
            else: selected.append(path)
        self.paths = selected
        self.tracked=set(self.paths)
        for p in self.paths:
            if p.suffix.lower() not in IMAGE_SUFFIXES and p.suffix.lower()!='.html':
                self.file_ids[p]='f'+hashlib.sha256(str(p).encode()).hexdigest()[:16]

    def heading(self,title,level,identifier=None,toc=True,pagebreak=False):
        ident=identifier or 'h'+str(len(self.entries))
        if pagebreak:self.blocks.append({'t':'RawBlock','c':['openxml','<w:p><w:r><w:br w:type="page"/></w:r></w:p>']})
        self.blocks.append({'t':'Header','c':[level,[ident,[],[]],[{'t':'Str','c':title}]]})
        if toc:self.entries.append({'title':title,'level':level,'id':ident,'page':1})
        return ident

    def image(self,path,caption=''):
        p=Path(path)
        if not p.is_file():raise FileNotFoundError(f'Unresolved image: {path}')
        self.used_images.add(p.resolve()); self.image_count+=1
        with Image.open(p) as im:
            w,h=im.size
            width=min(6.85,w/115,6.7*w/h)
        self.blocks.append({'t':'Para','c':[{'t':'Image','c':[['',[],[['width',f'{width:.3f}in']]],
            [{'t':'Str','c':caption or p.name}],[str(p),'']]}]})
        if caption and not re.fullmatch(r'(image|.*\.(png|jpg|jpeg))',caption,flags=re.I):
            self.blocks.append(para(caption,'Caption'))
        self.stats['images']+=1

    def markdown(self,s,path):
        ast=json.loads(subprocess.check_output(['pandoc','-f','markdown+tex_math_dollars+pipe_tables-yaml_metadata_block-simple_tables-multiline_tables-grid_tables','-t','json'],input=s.encode()))
        def walk(x):
            if isinstance(x,list):
                for a in x:walk(a)
            elif isinstance(x,dict):
                if x.get('t')=='Header':
                    x['c'][0]=min(6,x['c'][0]+3)
                    x['c'][1][0]=self.file_ids.get(path,'src')+'-'+x['c'][1][0]
                elif x.get('t')=='CodeBlock':
                    x['c'][1]=code(x['c'][1])['c'][1][0]['c'][1]
                elif x.get('t')=='Image':
                    target=x['c'][2][0]
                    if target.startswith('attachment:'):
                        target=str(self.attachments[target[11:]])
                    elif target.startswith('data:'):
                        mime,body=target.split(',',1); ext='.png' if 'png' in mime else '.jpg'
                        out=self.work/f'inline_{self.image_count}{ext}'; out.write_bytes(base64.b64decode(body));target=str(out)
                    elif not re.match(r'https?://',target):target=str((self.root/path.parent/urllib.parse.unquote(target)).resolve())
                    else:raise RuntimeError(f'Remote image requires a local asset: {target}')
                    p=Path(target)
                    if not p.exists():raise FileNotFoundError(f'{path}: {target}')
                    self.used_images.add(p.resolve());self.image_count+=1;self.stats['images']+=1
                    with Image.open(p) as im:w,h=im.size
                    width=min(6.85,w/115,6.7*w/h)
                    x['c'][2][0]=str(p);x['c'][0][2].append(['width',f'{width:.3f}in'])
                elif x.get('t')=='Link':
                    target=x['c'][2][0]
                    if not re.match(r'[a-zA-Z]+:',target) and not target.startswith('#'):
                        rp=(self.root/path.parent/urllib.parse.unquote(target.split('#')[0])).resolve()
                        try:rel=rp.relative_to(self.root)
                        except ValueError:rel=None
                        if rel in self.file_ids:x['c'][2][0]='#'+self.file_ids[rel]
                        else:x['c'][2][0]='https://github.com/Idan-Alter/OU-22971-Toolbox/blob/'+self.commit+'/'+str(rel or target)
                if 'c' in x:walk(x['c'])
        walk(ast['blocks']);self.blocks.extend(ast['blocks'])
        self.stats['markdown_sections']+=1
        self.api=ast['pandoc-api-version']

    def notebook(self,path):
        nb=json.loads((self.root/path).read_text(encoding='utf-8')); self.stats['notebooks']+=1
        for number,cell in enumerate(nb['cells'],1):
            s=source_text(cell.get('source',''));typ=cell['cell_type']
            self.attachments={}
            for name,representations in cell.get('attachments',{}).items():
                mime=next(iter(representations)); out=self.work/('attachment_'+str(self.image_count)+'_'+Path(name).name)
                out.write_bytes(base64.b64decode(source_text(representations[mime])));self.attachments[name]=out
            if typ=='markdown':self.markdown(s,path)
            elif typ=='raw':self.blocks.append(code(s));self.stats['raw_cells']+=1
            else:
                execution=cell.get('execution_count')
                self.blocks.append(para(f'Cell {number}  Input [{execution if execution is not None else " "}]','Block Label'))
                self.blocks.append(code(s));self.stats['code_cells']+=1
                for oi,out in enumerate(cell.get('outputs',[])):
                    self.stats['notebook_outputs']+=1
                    self.blocks.append(para('Output'+('  '+out.get('name','') if out['output_type']=='stream' else ''),'Block Label'))
                    if out['output_type']=='stream':self.blocks.append(code(source_text(out.get('text','')),True))
                    elif out['output_type']=='error':self.blocks.append(code('\n'.join(out.get('traceback',[])) or out.get('ename','')+': '+out.get('evalue',''),True))
                    else:
                        data=out.get('data',{})
                        if 'image/png' in data or 'image/jpeg' in data:
                            mime='image/png' if 'image/png' in data else 'image/jpeg'
                            p=self.work/f'output_{self.image_count}.{"png" if mime.endswith("png") else "jpg"}'
                            p.write_bytes(base64.b64decode(source_text(data[mime])));self.image(p)
                        elif 'text/html' in data and ('<table' in source_text(data['text/html']).lower() or 'text/plain' not in data) and (max([len(re.findall(r'<t[hd]\b', row, re.I)) for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', source_text(data['text/html']), re.S|re.I)] or [0]) <= 8 or 'text/plain' not in data):
                            # Strip web-only scripts/styles; preserve rendered tables and prose.
                            h=source_text(data['text/html']);h=re.sub(r'<svg\b.*?</svg>', '', h, flags=re.S|re.I);h=re.sub(r'<img\b[^>]*>', '', h, flags=re.I);h=re.sub(r'<(script|style)\b[^>]*>.*?</\1>','',h,flags=re.S|re.I)
                            a=json.loads(subprocess.check_output(['pandoc','-f','html','-t','json'],input=h.encode()))
                            if a['blocks']:self.blocks.extend(a['blocks'])
                            elif 'text/plain' in data:self.blocks.append(code(source_text(data['text/plain']),True))
                        elif 'text/plain' in data:self.blocks.append(code(source_text(data['text/plain']),True))
                        elif 'application/json' in data:self.blocks.append(code(json.dumps(data['application/json'],ensure_ascii=False,indent=2),True))
                        else:
                            self.blocks.append(code(json.dumps(data,ensure_ascii=False,indent=2),True))
                            self.warnings.append(f'{path} cell {number}: no standard display MIME; retained serialized output')

    def file(self,path):
        if path in self.audit:return
        p=self.root/path;suffix=p.suffix.lower()
        if suffix in IMAGE_SUFFIXES:return
        if suffix=='.html' and path.with_suffix('.ipynb') in self.tracked:
            self.audit[str(path)]={'treatment':'companion HTML represented by notebook and embedded original'};return
        self.heading(label(path),3,self.file_ids.get(path),toc=True)
        self.blocks.append(para('Source '+str(path),'Source Path'))
        self.stats['files_rendered']+=1
        if suffix=='.md':
            text=p.read_text(encoding='utf-8-sig')
            if path == Path('README.md'):text=re.split(r'^## Maintenance\s*$',text,flags=re.M)[0]
            self.markdown(text,path);mode='formatted markdown'
        elif suffix=='.ipynb':self.notebook(path);mode='notebook cells and saved outputs'
        elif suffix=='.html':
            a=json.loads(subprocess.check_output(['pandoc','-f','html','-t','json',str(p)]));self.blocks.extend(a['blocks']);mode='rendered HTML'
        elif suffix in TEXT_SUFFIXES or p.name in {'Dockerfile','.gitignore','.gitattributes','.dockerignore','.gitkeep'}:
            raw=p.read_bytes()
            text=raw.decode('utf-16' if raw.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8-sig')
            self.blocks.append(para('Saved output' if suffix=='.log' else 'Source code and configuration','Block Label'))
            self.blocks.append(code(text,suffix=='.log'));mode='complete output text' if suffix=='.log' else 'complete source code or configuration'
        else:
            # Fail rather than silently lose an unknown material type.
            raise RuntimeError(f'No readable renderer for {path}')
        self.audit[str(path)]={'treatment':mode}

    def directory(self,dirpath,level=2,title=None):
        name=title or label(dirpath)
        self.heading(name,level,toc=True,pagebreak=(level==1))
        children=set()
        for p in self.paths:
            if p==dirpath or dirpath not in p.parents:continue
            rel=p.relative_to(dirpath)
            children.add(dirpath/rel.parts[0])
        def priority(p):
            if p.name.lower()=='readme.md':return (0,())
            if p.name in {'environment.yml','environment.yaml','.devcontainer','.dockerignore','Dockerfile','docker-compose.yml'}:return (1,natural(p.name))
            if re.match(r'^\d',p.name):return (2,natural(p.name))
            if p in self.tracked:return (3,natural(p.name))
            return (4,natural(p.name))
        for p in sorted(children,key=priority):
            if p in self.tracked:self.file(p)
            elif p.name=='images':continue
            elif any(q==p or p in q.parents for q in self.paths):
                self.directory(p,min(2,level+1))

    def build(self):
        readme=Path('README.md');self.heading('Course overview',1,pagebreak=False)
        self.file(readme)
        s=(self.root/readme).read_text()
        parts=[]
        for link in re.findall(r'\]\(([^)]+)\)',s):
            p=Path(link)
            if len(p.parts)>1 and p.parts[0] not in parts and not re.match(r'[a-z]+:',link):parts.append(p.parts[0])
        roots={p.parts[0] for p in self.paths if len(p.parts)>1}
        parts=[p for p in parts if p in roots and p!='repo_maintenance']+sorted(roots-set(parts)-{'repo_maintenance'},key=natural)
        for i,part in enumerate(parts,1):self.directory(Path(part),1,f'Part {i}  '+label(Path(part)))
        if 'repo_maintenance' in roots:self.directory(Path('repo_maintenance'),1,'Appendix  Repository maintenance')
        leftovers=[p for p in self.paths if p not in {readme} and len(p.parts)==1 and p.suffix not in IMAGE_SUFFIXES]
        if leftovers:
            self.heading('Appendix  Repository configuration',1,pagebreak=True)
            for p in sorted(leftovers,key=natural):self.file(p)
        unused=[p for p in self.paths if p.suffix.lower() in IMAGE_SUFFIXES and (self.root/p).resolve() not in self.used_images]
        if unused:
            self.heading('Appendix  Additional figures',1,pagebreak=True)
            for p in unused:self.blocks.append(para(str(p),'Source Path'));self.image(self.root/p,label(p))
        for p in self.paths:
            if p.suffix.lower() in IMAGE_SUFFIXES:self.audit[str(p)]={'treatment':'image in context' if p not in unused else 'additional figure'}
        missing=set(map(str,self.paths))-set(self.audit)
        if missing:raise AssertionError(f'Unaccounted source files: {missing}')
        return {'pandoc-api-version':self.api,'meta':{},'blocks':self.blocks}

def set_font(style,name,size):
    style.font.name=name;style.font.size=Pt(size);style.font.color.rgb=RGBColor(0,0,0)
    rpr=style.element.get_or_add_rPr();fonts=rpr.find(qn('w:rFonts'))
    if fonts is not None:
        for key in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme']:fonts.attrib.pop(qn('w:'+key),None)
        for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),name)

def field(p,instruction,value=''):
    for kind,txt in [('begin',''),('instr',instruction),('separate',''),('text',value),('end','')]:
        r=OxmlElement('w:r')
        if kind in {'begin','separate','end'}:
            e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),kind)
        elif kind=='instr':e=OxmlElement('w:instrText');e.set(qn('xml:space'),'preserve');e.text=txt
        else:e=OxmlElement('w:t');e.text=txt
        r.append(e);p._p.append(r)

def style_document(path,exporter):
    doc=Document(path)
    for st in doc.styles:
        if re.match(r'^Heading [1-9]$', st.name): st.element.find(qn('w:name')).set(qn('w:val'), st.name.lower())
        elif st.name == 'TOC Heading': st.element.find(qn('w:name')).set(qn('w:val'), 'TOC Heading')
    sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.7);sec.bottom_margin=Inches(.65);sec.left_margin=Inches(.8);sec.right_margin=Inches(.8)
    sec.header_distance=Inches(.25);sec.footer_distance=Inches(.25);sec.different_first_page_header_footer=True
    set_font(doc.styles['Normal'],'DejaVu Sans',10)
    normal=doc.styles['Normal'].paragraph_format;normal.space_after=Pt(6);normal.line_spacing=1.13
    set_font(doc.styles['Title'],'DejaVu Sans',28)
    for i,size in enumerate([20,16,13,11.5,10.5,10],1):
        st=doc.styles[f'Heading {i}'];set_font(st,'DejaVu Sans',size)
        st.paragraph_format.keep_with_next=True;st.paragraph_format.space_before=Pt(12);st.paragraph_format.space_after=Pt(6)
    for name in ['Code Block','Output Block','Source Code','Verbatim Char']:
        if name not in doc.styles:doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        set_font(doc.styles[name],'DejaVu Sans Mono',8.2)
        if doc.styles[name].type==WD_STYLE_TYPE.PARAGRAPH:
            pf=doc.styles[name].paragraph_format;pf.space_before=Pt(4);pf.space_after=Pt(6);pf.line_spacing=1.05
            pf.keep_together=False;pf.keep_with_next=False;pf.left_indent=Inches(.07);pf.right_indent=Inches(.07)
    for name,size in [('Block Label',8.5),('Source Path',8),('Caption',8.5)]:
        if name not in doc.styles:doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        set_font(doc.styles[name],'DejaVu Sans',size);doc.styles[name].paragraph_format.keep_with_next=(name!='Caption')
    previous_label = ''
    for p in doc.paragraphs:
        if p.style.name == 'Source Code': p.style = doc.styles['Output Block' if previous_label.startswith('Output') else 'Code Block']
        if p.style.name == 'Block Label': previous_label = p.text
        elif p.style.name not in {'Code Block', 'Output Block'}: previous_label = ''
        if p.style.name in {'Code Block','Output Block','Source Code'}:
            # Paragraph shading, not one-cell tables: long code blocks can paginate.
            fill='F1F3F5' if p.style.name!='Output Block' else 'F8F8F8'
            shd=OxmlElement('w:shd');shd.set(qn('w:fill'),fill);p._p.get_or_add_pPr().append(shd)
            p.paragraph_format.keep_together=False;p.paragraph_format.keep_with_next=False
            for run in p.runs:run.font.name='DejaVu Sans Mono';run.font.size=Pt(8.2)
        if p._p.xpath('.//w:drawing'):
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_together=True
        for run in p.runs:
            if p.style.name.startswith('Heading') or p.style.name=='Title':run.font.color.rgb=RGBColor(0,0,0)
    for table in doc.tables:
        table.autofit=False
        for row in table.rows:
            for cell in row.cells:
                tcpr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
                for side in ['top','left','bottom','right','insideH','insideV']:
                    e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
                tcpr.append(borders)
                for p in cell.paragraphs:
                    p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3)
                    for r in p.runs:r.font.size=Pt(8)
        if table.rows:
            repeat=OxmlElement('w:tblHeader');table.rows[0]._tr.get_or_add_trPr().append(repeat)
            for cell in table.rows[0].cells:
                shd=OxmlElement('w:shd');shd.set(qn('w:fill'),'E8ECF0');cell._tc.get_or_add_tcPr().append(shd)
    # Front matter is inserted in document order before the first source heading.
    first=doc.paragraphs[0]
    def front(text='',style=None):return first.insert_paragraph_before(text,style)
    p=front('Toolbox for Machine Learning\nand Big Data','Title');p.paragraph_format.space_before=Pt(115);p.paragraph_format.space_after=Pt(24)
    front('Course 22971','Subtitle');front('Open University of Israel')
    p=front('Complete course reader');p.paragraph_format.space_before=Pt(26)
    front('MLOps with MLflow\nDistributed computing with Ray\nDistributed deep learning with PyTorch')
    p=front('Repository edition '+exporter.commit[:12]);p.paragraph_format.space_before=Pt(24)
    front('Notebook code and saved results appear in their original cell order. Course scripts, configuration, and figures are included.')
    p=front();p.add_run().add_break(WD_BREAK.PAGE)
    p=front('Contents','TOC Heading' if 'TOC Heading' in doc.styles else None)
    set_font(p.style,'DejaVu Sans',18)
    for index,entry in enumerate(exporter.entries):
        name=f'TOC {min(entry["level"],3)}'
        if name not in doc.styles:doc.styles.add_style(name,WD_STYLE_TYPE.PARAGRAPH)
        set_font(doc.styles[name],'DejaVu Sans',9 if entry['level']==3 else 10)
        p=front('',name);p.paragraph_format.space_after=Pt(3);p.paragraph_format.keep_with_next=False
        p.paragraph_format.left_indent=Inches((entry['level']-1)*.15)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(6.8-(entry['level']-1)*.15),WD_TAB_ALIGNMENT.RIGHT)
        if index==0:
            r=OxmlElement('w:r');e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'begin');r.append(e);p._p.append(r)
            r=OxmlElement('w:r');e=OxmlElement('w:instrText');e.set(qn('xml:space'),'preserve');e.text=' TOC \\o "1-3" \\h \\z \\u ';r.append(e);p._p.append(r)
            r=OxmlElement('w:r');e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'separate');r.append(e);p._p.append(r)
        h=OxmlElement('w:hyperlink');h.set(qn('w:anchor'),entry['id'])
        r=OxmlElement('w:r');t=OxmlElement('w:t');t.text=entry['title'];r.append(t);h.append(r);p._p.append(h)
        p.add_run('\t');field(p,' PAGEREF '+entry['id']+' \\h ','1')
        p._p.set(qn('w:rsidR'),f'{index+1:08X}') # stable TOC entry identity
        if index==len(exporter.entries)-1:
            r=OxmlElement('w:r');e=OxmlElement('w:fldChar');e.set(qn('w:fldCharType'),'end');r.append(e);p._p.append(r)
    p=front();p.add_run().add_break(WD_BREAK.PAGE)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    set_font(footer.style,'DejaVu Sans',8);field(footer,' PAGE ','1')
    update=doc.settings.element.find(qn('w:updateFields'))
    if update is not None:doc.settings.element.remove(update)
    doc.core_properties.title='Toolbox for Machine Learning and Big Data Course 22971'
    doc.core_properties.subject='Complete course repository reader';doc.core_properties.author='Open University of Israel'
    doc.save(path)

def paginate(path,entries,work):
    import fitz
    soffice=os.environ.get('SOFFICE') or shutil.which('soffice')
    if not soffice:
        candidate=Path(__import__('sys').executable).parents[2]/'bin/override/soffice'
        if candidate.exists():soffice=str(candidate)
    if not soffice:raise RuntimeError('--paginate requires LibreOffice (soffice)')
    previous=None
    for iteration in range(5):
        out=work/f'pagination_{iteration}';out.mkdir()
        subprocess.run([soffice,'-env:UserInstallation='+ (work/'lo_profile').as_uri(),'--headless','--convert-to','pdf','--outdir',str(out),str(path)],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
        pdf=fitz.open(out/(path.stem+'.pdf'))
        # Match heading paragraph lines in body pages; ignore contents pages.
        pages={};texts=[' '.join(p.get_text().split()) for p in pdf]
        first_body=next(i for i,t in enumerate(texts) if 'Source README.md' in t)
        for entry in entries:
            needle=' '.join(entry['title'].split())
            for i in range(first_body,len(pdf)):
                # File headings disambiguated by full source path below them.
                marker=entry.get('source_marker')
                if (marker and marker in re.sub(r'\s+','',texts[i])) or (not marker and needle in texts[i]):
                    pages[entry['id']]=i+1;break
            if entry['id'] not in pages:raise RuntimeError('Cannot locate TOC heading: '+entry['title'])
        doc=Document(path)
        toc=[p for p in doc.paragraphs if p.style.name.startswith('TOC ') and p.style.name!='TOC Heading']
        if len(toc)!=len(entries):raise AssertionError('TOC entries changed during pagination')
        for p,entry in zip(toc,entries):
            results=p._p.xpath('.//w:t');results[-1].text=str(pages[entry['id']]);entry['page']=pages[entry['id']]
        doc.save(path)
        print(f'Pagination {iteration+1}: {len(pdf)} pages',flush=True)
        pdf.close()
        if pages==previous:return
        previous=pages
    raise RuntimeError('TOC page numbers did not stabilize after five passes')

def embed_sources(path,exporter):
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for rel in exporter.paths:z.write(exporter.root/rel,str(rel))
        z.writestr('COURSE_EXPORT_MANIFEST.json',json.dumps({'commit':exporter.commit,'files':exporter.audit},indent=2))
    with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as z:
        z.writestr(ARCHIVE_PART,archive.getvalue())
        # A custom package relationship keeps the archive reachable in OPC.
        rels=ET.fromstring(z.read('_rels/.rels'));ns='http://schemas.openxmlformats.org/package/2006/relationships'
        ET.SubElement(rels,'{'+ns+'}Relationship',Id='rIdCourseSources',Type='urn:course-export:source-archive',Target=ARCHIVE_PART)
        content=ET.fromstring(z.read('[Content_Types].xml'));cns='http://schemas.openxmlformats.org/package/2006/content-types'
        ET.SubElement(content,'{'+cns+'}Override',PartName='/'+ARCHIVE_PART,ContentType='application/zip')
        # Replace XML entries without duplicate ZIP names.
        replacements={'_rels/.rels':ET.tostring(rels,encoding='utf-8',xml_declaration=True),'[Content_Types].xml':ET.tostring(content,encoding='utf-8',xml_declaration=True)}
    tmp=path.with_suffix('.tmp')
    with zipfile.ZipFile(path) as src,zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as dst:
        for info in src.infolist():dst.writestr(info,replacements.get(info.filename,src.read(info.filename)))
    tmp.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('repo',nargs='?',type=Path);parser.add_argument('--output',type=Path,default=Path('course_22971.docx'))
    parser.add_argument('--paginate',action='store_true');parser.add_argument('--embed-sources',action='store_true',help='Include byte-exact originals in an extractable source archive')
    parser.add_argument('--extract-sources',type=Path);parser.add_argument('--destination',type=Path,default=Path('extracted_course_sources'))
    args=parser.parse_args()
    if args.extract_sources:
        args.destination.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(args.extract_sources) as z,zipfile.ZipFile(io.BytesIO(z.read(ARCHIVE_PART))) as src:
            for name in src.namelist():
                target=(args.destination/name).resolve()
                if not target.is_relative_to(args.destination.resolve()):raise ValueError('Unsafe archive path')
            src.extractall(args.destination)
        print('Extracted original repository to '+str(args.destination));return
    if not args.repo:parser.error('repo path is required')
    root=args.repo.resolve();output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='course_docx_') as tmp:
        work=Path(tmp);exporter=Exporter(root,work);ast=exporter.build()
        ap=work/'course.json';ap.write_text(json.dumps(ast,ensure_ascii=False),encoding='utf-8')
        subprocess.run(['pandoc','-f','json','-t','docx','--highlight-style=monochrome',str(ap),'-o',str(output)],check=True)
        style_document(output,exporter)
        for entry in exporter.entries:
            for rel,ident in exporter.file_ids.items():
                if ident==entry['id']:entry['source_marker']='Source'+re.sub(r'\s+','',str(rel))
        if args.paginate:paginate(output,exporter.entries,work)
        for rel in exporter.paths:
            exporter.audit[str(rel)]['sha256']=hashlib.sha256((root/rel).read_bytes()).hexdigest()
            exporter.audit[str(rel)]['size_bytes']=(root/rel).stat().st_size
        if args.embed_sources:embed_sources(output,exporter)
        report={'repository':str(root),'commit':exporter.commit,'stats':dict(exporter.stats),
            'included_files':len(exporter.paths),'excluded_files':exporter.excluded,'render_order':[e for e in exporter.entries],
            'warnings':exporter.warnings,'files':exporter.audit,'embedded_sources':args.embed_sources}
        output.with_suffix('.audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'output':str(output),'tracked_files':len(exporter.paths),'stats':dict(exporter.stats),'warnings':exporter.warnings},indent=2))

if __name__=='__main__':main()
