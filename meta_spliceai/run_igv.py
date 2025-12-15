import igv


def demo(): 

    input_path = "/Users/barnettchiu/work/meta-spliceai/data/ensembl/h38.final.bed"

    # Initialize the IGV browser
    b = igv.Browser({
        "genome": "hg38",  # Reference genome (e.g., hg19, hg38, mm10)
        "locus": "chr1:1000000-1010000",  # Initial locus to view
        "tracks": [
            {
                "name": "Example BED Track",
                "type": "bed",
                "url": input_path,  # URL or path to your BED file
                "color": "rgb(0,100,0)"
            }
        ]
    })

    # Display the IGV browser
    b.show()


if __name__ == "__main__":
    demo()  
