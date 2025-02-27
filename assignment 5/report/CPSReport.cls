%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Chair of Cyber-Physical-Systems
% Univ.-Prof. Dr. Elmar Rueckert
% Montanuniversität Leoben, Austria
% Latest Update: March 2022
%
% Disclaimer: The materials and source code are for personal use only. The material 
% is intended for educational purposes only. Reproduction of the material for any 
% purposes other than what is intended is prohibited. The content is to be used for 
% educational and non-commercial purposes only and is not to be changed, altered, 
% or used for any commercial endeavor without the express written permission of 
% Professor Rueckert. 
% 
% Parts of this latex template rely on an template by Frits Wenneker, 28/2/17, 
%  License: CC BY-NC-SA 3.0 (http://creativecommons.org/licenses/by-nc-sa/3.0/)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}[1994/06/01]
\ProvidesPackage{CPSReport}[2022/03/31 CPS Student Report Template by Univ.-Prof. Dr. Elmar Rueckert]
%\LoadClass[11pt, a4paper, twoside]{article}
\LoadClass[11pt, a4paper, twocolumn]{article}
%----------------------------------------------------------------------------------------
%	PACKAGES AND OTHER DOCUMENT CONFIGURATIONS
%----------------------------------------------------------------------------------------

\usepackage[english]{babel} % English language hyphenation

\usepackage{microtype} % Better typography

\usepackage{amsmath,amsfonts,amsthm} % Math packages for equations

\usepackage[svgnames]{xcolor} % Enabling colors by their 'svgnames'

\usepackage[hang, small, labelfont=bf, up, textfont=it]{caption} % Custom captions under/above tables and figures

\usepackage{booktabs} % Horizontal rules in tables

\usepackage{lastpage} % Used to determine the number of pages in the document (for "Page X of Total")

\usepackage{graphicx} % Required for adding images

\usepackage{enumitem} % Required for customising lists
\setlist{noitemsep} % Remove spacing between bullet/numbered list elements

\usepackage{sectsty} % Enables custom section titles
\allsectionsfont{\usefont{OT1}{phv}{b}{n}} % Change the font of all section commands (Helvetica)

\usepackage{listings} % To print python code
%----------------------------------------------------------------------------------------
%	MATH COMMANDS
%----------------------------------------------------------------------------------------
%\input{math_commands.tex}


%----------------------------------------------------------------------------------------
%	Corporate Design Definitions
%----------------------------------------------------------------------------------------
\definecolor{MULTurquoise}{rgb}{0.0, 0.45, 0.49}
\definecolor{MULSmoke}{rgb}{0.23, 0.22, 0.22}

\definecolor{mygreen}{RGB}{28,172,0} % color values Red, Green, Blue
\definecolor{mylilas}{RGB}{170,55,241}

%[1]{{\color{MULTurquoise}#1}} % Authors style (Helvetica)
%\large\usefont{OT1}{phv}{b}{n}
%----------------------------------------------------------------------------------------
%	MARGINS AND SPACING
%----------------------------------------------------------------------------------------

\usepackage{geometry} % Required for adjusting page dimensions

\geometry{
	top=1cm, % Top margin
	bottom=1.5cm, % Bottom margin
	left=2cm, % Left margin
	right=2cm, % Right margin
	includehead, % Include space for a header
	includefoot, % Include space for a footer
	%showframe, % Uncomment to show how the type block is set on the page
}

\setlength{\columnsep}{7mm} % Column separation width

%----------------------------------------------------------------------------------------
%	FONTS
%----------------------------------------------------------------------------------------

\usepackage[T1]{fontenc} % Output font encoding for international characters
\usepackage[utf8]{inputenc} % Required for inputting international characters

\usepackage{XCharter} % Use the XCharter font

%----------------------------------------------------------------------------------------
%	HEADERS AND FOOTERS
%----------------------------------------------------------------------------------------

\usepackage{fancyhdr} % Needed to define custom headers/footers
\pagestyle{fancy} % Enables the custom headers/footers

\renewcommand{\headrulewidth}{0.0pt} % No header rule
\renewcommand{\footrulewidth}{0.4pt} % Thin footer rule

\renewcommand{\sectionmark}[1]{\markboth{#1}{}} % Removes the section number from the header when \leftmark is used

%\nouppercase\leftmark % Add this to one of the lines below if you want a section title in the header/footer

% Headers
\lhead{} % Left header
\chead{\textit{\thetitle}} % Center header - currently printing the article title
\rhead{} % Right header

% Footers
\lfoot{} % Left footer
\cfoot{} % Center footer
%\rfoot{\footnotesize Page \thepage\ of \pageref{LastPage}} % Right footer, "Page 1 of 2"
\rfoot{\color{MULTurquoise} {\footnotesize Page \thepage\ of \pageref{LastPage}}}

\fancypagestyle{firstpage}{ % Page style for the first page with the title
	\fancyhf{}
	%\renewcommand{\footrulewidth}{0pt} % Suppress footer rule
	\rfoot{\color{MULTurquoise} {\footnotesize Page \thepage\ of \pageref{LastPage}}}
}

\fancypagestyle{lastpage}{ % Page style for the first page with the title
	\fancyhf{}
	\lfoot{\color{MULTurquoise} {\footnotesize Template by Chair of Cyber-Physical-Systems, Montanuniversität Leoben, Austria}}
	\rfoot{\color{MULTurquoise} {\footnotesize Page \thepage\ of \pageref{LastPage}}}
	%\renewcommand{\footrulewidth}{0pt} % Suppress footer rule
} 

%----------------------------------------------------------------------------------------
%	TITLE SECTION
%----------------------------------------------------------------------------------------
%----------------------------------------------------------------------------------------
%	Chai of Cyber-Physical-Systems Commands
%----------------------------------------------------------------------------------------
\newcommand{\submissiondate}[1]{#1}

\newcommand{\coursetitle}[1]{{\large\usefont{OT1}{phv}{b}{n}\color{MULTurquoise}#1}\newline} % Authors style (Helvetica)

\newcommand{\authorstyle}[1]{{\large\usefont{OT1}{phv}{b}{n}\color{MULTurquoise}#1}} % Authors style (Helvetica)

\newcommand{\institution}[1]{{\footnotesize\usefont{OT1}{phv}{m}{sl}\color{Black}#1}} % Institutions style (Helvetica)

\usepackage{titling} % Allows custom title configuration

\newcommand{\HorRule}{\color{MULSmoke}\rule{\linewidth}{1pt}} % Defines the gold horizontal rule around the title

\pretitle{
	\vspace{-30pt} % Move the entire title section up
	\HorRule\vspace{10pt} % Horizontal rule before the title
	\fontsize{22}{36}\usefont{OT1}{phv}{b}{n}\selectfont % Helvetica
	\color{MULTurquoise} % Text colour for the title and author(s)
}

\posttitle{\par\vskip 15pt} % Whitespace under the title
%\course
\preauthor{} % Anything that will appear before \author is printed

\postauthor{ % Anything that will appear after \author is printed
	\vspace{5pt}
	\submissiondate
	\vspace{10pt} % Space before the rule
	\par\HorRule % Horizontal rule after the title
	\vspace{-25pt} % Space after the title section
}

%----------------------------------------------------------------------------------------
%	ABSTRACT
%----------------------------------------------------------------------------------------

\usepackage{lettrine} % Package to accentuate the first letter of the text (lettrine)
\usepackage{fix-cm}	% Fixes the height of the lettrine

\newcommand{\initial}[1]{ % Defines the command and style for the lettrine
	\lettrine[lines=3,findent=4pt,nindent=0pt]{% Lettrine takes up 3 lines, the text to the right of it is indented 4pt and further indenting of lines 2+ is stopped
		\color{MULSmoke}% Lettrine colour
		{#1}% The letter
	}{}%
}

\usepackage{xstring} % Required for string manipulation

\newcommand{\lettrineabstract}[1]{
	\StrLeft{#1}{1}[\firstletter] % Capture the first letter of the abstract for the lettrine
	\initial{\firstletter}\textbf{\StrGobbleLeft{#1}{1}} % Print the abstract with the first letter as a lettrine and the rest in bold
}

%----------------------------------------------------------------------------------------
%	BIBLIOGRAPHY
%----------------------------------------------------------------------------------------

\usepackage[backend=bibtex,style=authoryear,natbib=true]{biblatex} % Use the bibtex backend with the authoryear citation style (which resembles APA)

%\addbibresource{literature.bib} % The filename of the bibliography

\usepackage[autostyle=true]{csquotes} % Required to generate language-dependent quotes in the bibliography

\date{} 

%% My Packages
\usepackage{minted}
\usepackage{pgf}
\usepackage{float}

\usepackage{amsmath}
\usepackage[hidelinks]{hyperref}

\usepackage{blindtext}

\usepackage{placeins} % For FloatBarrier