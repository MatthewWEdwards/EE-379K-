import re
import sys
import math
import scipy as sp
import numpy as np
from bs4 import BeautifulSoup
import urllib
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
import collections

#download pdfs
if len(sys.argv) > 1 and sys.argv[1] == "1":
    with open("machine_learning.html") as fp:
    	soup = beautifulsoup(fp, 'html.parser')
    
    link_tags = soup.find_all("p", "links")
    
    pdf_tags = []
    for links in link_tags:
    	pdf_tags.append(links.contents[3]['href'])
    
    pdf_count = 0
    for pdf in pdf_tags:
    	pdf_opener = urllib.urlopener()
    	pdf_opener.retrieve(pdf, "./pdfs/" + str(pdf_count) + ".pdf")
    	pdf_count = pdf_count + 1

#read text from pdfs
words = {}
for pdf_num in range(0,434):
	fp = file("./pdfs/" + str(pdf_num) + ".pdf", 'rb')

	pagenums = set()
	output = StringIO()
	manager = PDFResourceManager()
	converter = TextConverter(manager, output, laparams=LAParams())
	interpreter = PDFPageInterpreter(manager, converter)

	for page in PDFPage.get_pages(fp, pagenums):
		interpreter.process_page(page)
		words.update(collections.Counter(re.findall(r'\w+', output.getvalue())))
	fp.close()

word_counts = open("word_freq.txt", "w")
word_counts.write(str(words))
print words




	



