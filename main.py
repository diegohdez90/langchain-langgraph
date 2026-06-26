"""Main module for the program.
"""
from dotenv import load_dotenv
from getpass import getpass
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic as Anthropic

load_dotenv()


INFORMATION = """
Dennis M. Ritchie (born September 9, 1941, Bronxville, Eastchester,
New York, U.S.—found dead October 2011, Berkeley Heights, New Jersey)
was an American computer scientist and co-winner of the 
1983 A.M. Turing Award, the highest honour in computer science.
Ritchie and the American computer scientist Kenneth L. Thompson were
cited jointly for “their development of generic operating systems
theory and specifically for the implementation of the UNIX operating
system,” which they developed together at Bell Laboratories.

Ritchie earned a bachelor’s degree (1963) in physics and a
doctorate (1968) in mathematics from Harvard University. In 1967 he
joined Bell Labs, where he first worked on the Multics operating
system (OS). Multics was a time-sharing system funded by the Advanced
Research Projects Agency and jointly developed by researchers at the
Massachusetts Institute of Technology, Bell Labs, and General
Electric Co. However, AT&T Corporation (then the parent company of
Bell Labs) withdrew from the project and removed its GE computers in 1969.

Upon the removal of the GE machines, Ritchie joined Thompson in developing
a more flexible operating system for Bell Lab’s obsolete Digital Equipment
Corporation (DEC) PDP-7 minicomputer. Within a few months they had created
UNIX, a new OS not completely tied to any particular computer hardware,
as earlier systems had been.

In conjunction with the development of UNIX, Ritchie contributed somewhat
to Thompson’s creation of the B programming language in 1970. As they
moved their operating system to a newer PDP-11 minicomputer in 1971, the
shortcomings of B became apparent, and Ritchie extended the language over
the next year to create the C programming language. C and its family of
languages, including C++ and Java, remain among the most widely used
computer programming languages. In 1973 Ritchie and Thompson rewrote UNIX in C.

Ritchie was named a fellow by Bell Labs in 1983 and was elected to the U.S.
National Academy of Engineering in 1988. In 1990 he was appointed head of the
System Software Research Department at Bell Labs, where he led the development
of the Plan 9 (1995) and Inferno (1996) operating systems. In 1998 Ritchie and
Thompson were awarded the U.S. National Medal of Technology for their
developmentof UNIX.
"""

QUESTION = f"""
Given the information {INFORMATION} I want you create:
1. A short summary of the information provided.
2. 3 interesting facts about Dennis M. Ritchie.
"""


def main():
    """Main function to execute the program.
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["INFORMATION"],
        template=QUESTION
    )
    
    llm = Anthropic(model="claude-2", temperature=0.7)
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"INFORMATION": INFORMATION})
    print("Summary:")
    print(response.content)


if __name__ == '__main__':
    main()
