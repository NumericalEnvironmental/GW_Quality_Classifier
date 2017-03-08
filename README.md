# GW_Quality_Classifier
This is python 2.7 script which reads, processes, and evaluates a large, multi-county groundwater quality data set. The script employs the following libraries:

* pandas, to read, merge, and pivot the data
* matplotlib, to create histograms
* scikit-learn, to classify data points with similar changes in concentration with time for a particular constituent (e.g., chloride)
* scipy.stats class, to conduct various statistical analyses

The script is hard-wired to read and merge individual county data files from the California State Water Resources Control Board’s Groundwater Ambient Monitoring and Assessment Program (GAMA). GAMA is an online (http://geotracker.waterboards.ca.gov/gama/), publicly-accessible repository for groundwater water quality for both environmental monitoring wells and water supply wells, including municipal wells, private domestic wells, and agricultural wells. Kern, Kings, Tulare and Fresno counties were all included in an application of the script. Because the data files associated with these counties are large (exceeding 100 MB in two cases), users should download the data files separately from the GAMA access portal (http://geotracker.waterboards.ca.gov/gama/datadownload).

More background information can be found here: https://numericalenvironmental.wordpress.com/2017/03/08/another-python-script-for-exploring-a-multiparameter-groundwater-quality-data-set-san-joaquin-valley-ca/

Email me with questions at walt.mcnab@gmail.com.

THIS CODE/SOFTWARE IS PROVIDED IN SOURCE OR BINARY FORM "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

