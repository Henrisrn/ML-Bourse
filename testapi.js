

    const sendPostRequest = async (url, data) => {
        try {
          const response = await fetch(url, {
            method: 'POST',
            headers: {
            },
            body: {
                'client_id': '3MVG99qusVZJwhskC.5R66GUqsKD1jfke_dgKieD8tblrMUBqBPmqPQFuUnYgZF9kkR7Z2_rexsEEzjGQXWgB',    
                'client_secret': 'DB9DB4774AD895E7B18FE3C43801D6C43D55547E6EB4C95AA558E288740FD3BB',
                'grant_type': 'password',
                'username': 'apiuser@amadeus.sb1',
                'password': '2@mbkybmMTB6JTZH60S8D6sAt2E6Udt3b'
                }
          });
          return await response.json();
        } catch (error) {
          console.error(error);
        }
      };
      
      const data = {
      };
      
      const result = await sendPostRequest('https://login.salesforce.com/services/oauth2/token', data);
      console.log(result);