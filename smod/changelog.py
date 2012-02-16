import git
import time

class DebChangelogFile(object):
    def __init__(self,reppath,ptag):
        self.repo = git.Repo(reppath)
        self.head = self.__get_head()
        self.tag = self.__get_ptag(ptag)
        self.__pkg_version = ""
        self.__pkg_name = ""
        self.__pkg_dist = ""
        self.__pkg_ulevel = ""
        self.__entry_header = ""

        self.eh_format = "%s (%s) %s; urgency=%s\n"
        self.eh = ""

    def __get_pkg_version(self):
        return self.__pkg_version

    def __set_pkg_version(self,version):
        self.__pkg_version = version
        self.__eh_update()

    version = property(__get_pkg_version,__set_pkg_version)

    def __get_pkg_name(self):
        return self.__pkg_name

    def __set_pkg_name(self,name):
        self.__pkg_name = name
        self.__eh_update()

    pkg_name = property(__get_pkg_name,__set_pkg_name)

    def __get_pkg_dist(self):
        return self.__pkg_dist

    def __set_pkg_dist(self,dist):
        self.__pkg_dist = dist
        self.__eh_update()

    distribution = property(__get_pkg_dist,__set_pkg_dist)

    def __get_pkg_ulevel(self):
        return self.__pkg_ulevel

    def __set_pkg_ulevel(self,level):
        self.__pkg_ulevel = level
        self.__eh_update()

    urgency = property(__get_pkg_ulevel,__set_pkg_ulevel)


    def __get_head(self):
        """
        __get_head(self):
        Returns the active head of the actual 
        GIT repository.
        """
        #get the active branch
        bname = self.repo.active_branch
        for h in self.repo.heads:
            if h.name ==bname:
                head = h
                break

        return head

    def __get_ptag(self,tname):
        """
        __get_ptag(self,tname):
        Returns the tag of the previous release.
        """
        #get the last tagged version
        for t in self.repo.tags:
            if t.name == tname:
                tag = t
                break

        return tag

    def __eh_update(self):
        self.__entry_header = self.eh_format %(self.pkg_name,self.version,self.distribution,self.urgency)

    def ce_footer(self,name,mail,date):
        date_format = "%a, %d %b %Y %H:%M:%S"
        date_str = time.strftime(date_format,date)

        tz_sign = "+" 
        if(time.timezone<0): tz_sign="-"
        tz_offset = abs(time.timezone)/60
        tz_offmin = tz_offset%60
        tz_offh = (tz_offset-tz_offmin)/60

        ostr = " -- %s <%s>  %s %s%02i%02i\n" %(name,mail,date_str,tz_sign,tz_offh,tz_offmin)
        return ostr

    def ce_text(self,text):
        text = text.replace("\n","\n    ")
        ostr = "  * %s\n" %text

        return ostr

    def create(self,clname):
        """
        create(clname):
        Create changelog file.
        """

        #iterate over all commits since the last version
        citer = self.repo.commits_between(self.tag.commit.id,self.head.commit.id)
        lc = []
        for c in citer:
            lc.append(c)

        lc.reverse()
            
        
        lfile = open(clname,"w")

        for c in lc:
            text = c.message
            author = c.author.name
            email = c.author.email
            date = c.committed_date
            text = self.ce_text(text)
            footer = self.ce_footer(author,email,date)
            lfile.write(self.__entry_header+"\n")
            lfile.write(text+"\n")
            lfile.write(footer+"\n")
    
        lfile.close()
    

